import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from typing import Annotated, Any, Optional, TypeVar
import traceback

import wx, wx.lib.scrolledpanel
from PIL import Image

from stable_diffusion import DEFAULT_OPTIONS, Direction, StableDiffusion, Status


IMAGE_SIZE = 512


SIZER = TypeVar('SIZER', bound=wx.Sizer)

class SizerStack:
    owner: wx.Window
    stack: list[wx.Sizer] = []

    def __init__(self, owner:wx.Window):
        self.owner = owner

    @property
    def top(self) -> wx.Sizer:
        if self.stack:
            return self.stack[-1]
        else:
            raise IndexError('Sizerがありません')

    def Add(self, item, proportion:Annotated[int, '引数flagが指定されていなかったら、flagになる']=0, flag:Optional[int]=None, border=3, userData=None, **kwargs):
        if flag is None:
            flag = proportion
            proportion = 0
        if border > 0:
            flag |= wx.ALL
        self.top.Add(item, proportion, flag, border, userData, **kwargs)

    @contextmanager
    def sizer(self, sizer:SIZER, **kwargs) -> Generator[SIZER, None, None]:
        try:
            is_root = not self.stack

            if is_root:
                self.owner.SetSizer(sizer)
                #if kwargs:
                #    LOG.warn('ルートsizerのオプションを**kwargsで指定することはできません')
            else:
                self.Add(sizer, **kwargs)

            self.stack.append(sizer)
            yield sizer

            if is_root:
                self.owner.SetAutoLayout(True)
                sizer.Fit(self.owner)
        finally:
            self.stack.pop()


class SDOptions:
    stable_diffusion: StableDiffusion
    controls: dict[str, wx.Control]

    def __init__(self, stable_diffusion:StableDiffusion, parent:wx.Window, sizers:SizerStack):
        self.stable_diffusion = stable_diffusion
        self.controls = {
            'prompt':          wx.TextCtrl(parent),
            'negative_prompt': wx.TextCtrl(parent),
            'steps':           wx.SpinCtrl(parent, min=1, max=200),
            'cfg_scale':       wx.SpinCtrl(parent, min=1, max=100),
            'mask_blur':       wx.SpinCtrl(parent, min=0, max=256),
            'sampler_name':    wx.ComboBox(parent),
            'scheduler':       wx.ComboBox(parent),
        }
        with sizers.sizer(wx.FlexGridSizer(len(self.controls), 2, 3, 3), flag=wx.EXPAND) as sizer:
            sizer.AddGrowableCol(1, 1)
            for key, ctrl in self.controls.items():
                sizers.Add(wx.StaticText(parent, label=key), flag=wx.ALIGN_CENTER_VERTICAL)
                sizers.Add(ctrl, 1, wx.EXPAND if isinstance(ctrl, wx.TextCtrl) else 0)

    async def fill_in_combo_box_choices(self):
        samplers   = await self.stable_diffusion.get_sampler_or_scheduler_names('samplers')
        schedulers = await self.stable_diffusion.get_sampler_or_scheduler_names('schedulers')
        def _do_update():
            self.controls['sampler_name'].Set(samplers)   # type: ignore
            self.controls['scheduler'   ].Set(schedulers) # type: ignore
        wx.CallAfter(_do_update)
        self.from_dict(DEFAULT_OPTIONS)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for key, ctrl in self.controls.items():
            out[key] = ctrl.Value # type: ignore
        return out

    def from_dict(self, options:dict[str, Any]):
        for key, value in options.items():
            control = self.controls.get(key, None)
            if control is not None:
                control.Value = value # type: ignore


class MainFrame(wx.Frame):
    scrolled_panel: wx.lib.scrolledpanel.ScrolledPanel
    canvas:         wx.Panel
    image:          Optional[Image.Image] = None
    bitmap:         Optional[wx.Bitmap] = None

    direction_buttons:      list[tuple[wx.RadioButton, Direction]]
    gen_width_control:      wx.SpinCtrl
    n_consecutive_control:  wx.SpinCtrl
    generate_cancel_button: wx.Button
    consecutive_gen_button: wx.Button
    progress_bar:           wx.Gauge

    sd_options:    SDOptions
    status:        Status = 'idle'
    is_horizontal: Optional[bool] = None

    stable_diffusion: StableDiffusion

    def __init__(self, stable_diffusion:StableDiffusion):
        super().__init__(None)
        self.stable_diffusion = stable_diffusion
        self.direction_buttons = []
    
        root_panel = wx.Panel(self)
        sizers = SizerStack(root_panel)

        with sizers.sizer(wx.BoxSizer(wx.VERTICAL)): # ルート
            with sizers.sizer(wx.BoxSizer(), border=6): # 画像上部
                # 方向指定
                with sizers.sizer(wx.GridSizer(3, 3, 1, 1)):
                    def _direction_button(dir:Direction) -> wx.RadioButton:
                        button = wx.RadioButton(root_panel)
                        self.direction_buttons.append((button, dir))
                        button.Bind(wx.EVT_BUTTON, lambda _: self.set_status())
                        return button
                    EMPTY = (0, 0)

                    sizers.Add(EMPTY, border=0)
                    sizers.Add(_direction_button(Direction.UP), flag=wx.RB_GROUP, border=0)
                    sizers.Add(EMPTY, border=0)

                    sizers.Add(_direction_button(Direction.LEFT), border=0)
                    sizers.Add(EMPTY, border=0)
                    sizers.Add(_direction_button(Direction.RIGHT), border=0)

                    sizers.Add(EMPTY, border=0)
                    sizers.Add(_direction_button(Direction.DOWN), border=0)
                    sizers.Add(EMPTY, border=0)

                    self.direction_buttons[2][0].Value = True

                sizers.Add(wx.StaticLine(root_panel, style=wx.LI_VERTICAL), 1, wx.EXPAND)

                # オプション
                with sizers.sizer(wx.BoxSizer(wx.VERTICAL)):
                    sizers.Add(wx.StaticText(root_panel, label='生成幅(px)'))
                    self.gen_width_control = wx.SpinCtrl(root_panel, initial=192, min=32, max=IMAGE_SIZE-32)
                    sizers.Add(self.gen_width_control)
                    self.gen_width_control.Increment = 32

                # 「生成」ボタン
                self.generate_cancel_button = wx.Button(root_panel, label='', size=wx.Size(100, 40))
                sizers.Add(self.generate_cancel_button, flag=wx.ALIGN_CENTER_VERTICAL)
                self.generate_cancel_button.Bind(wx.EVT_BUTTON, lambda _: asyncio.run_coroutine_threadsafe(self._generate_coroutine(), self.stable_diffusion.event_loop))
                self.generate_cancel_button.Enabled = False

                sizers.Add(wx.StaticLine(root_panel, style=wx.LI_VERTICAL), 1, wx.EXPAND)

                # 連続生成数
                with sizers.sizer(wx.BoxSizer(wx.VERTICAL)):
                    sizers.Add(wx.StaticText(root_panel, label='連続生成数'))
                    self.n_consecutive_control = wx.SpinCtrl(root_panel, initial=4, min=1, max=100)
                    sizers.Add(self.n_consecutive_control)

                # 「連続生成」ボタン
                self.consecutive_gen_button = wx.Button(root_panel, label='連続生成', size=wx.Size(100, 40))
                sizers.Add(self.consecutive_gen_button, flag=wx.ALIGN_CENTER_VERTICAL)
                self.consecutive_gen_button.Bind(wx.EVT_BUTTON, lambda _: asyncio.run_coroutine_threadsafe(self._generate_coroutine(self.n_consecutive_control.GetValue()), self.stable_diffusion.event_loop))
                self.consecutive_gen_button.Enabled = False

            # プログレスバー
            self.progress_bar = wx.Gauge(root_panel, range=100)
            sizers.Add(self.progress_bar, 0, border=6, flag=wx.EXPAND)

            self.sd_options = SDOptions(stable_diffusion, root_panel, sizers)
            print(self.sd_options.to_dict())

            self.scrolled_panel = wx.lib.scrolledpanel.ScrolledPanel(root_panel, size=wx.Size(IMAGE_SIZE, IMAGE_SIZE))
            self.canvas = wx.Panel(self.scrolled_panel, size=wx.Size(IMAGE_SIZE, IMAGE_SIZE))
            self.canvas.Bind(wx.EVT_PAINT, self._do_draw_image)
            self.canvas.MinSize = self.canvas.Size # type: ignore
            scr_sizer = wx.BoxSizer()
            scr_sizer.Add(self.canvas)
            self.scrolled_panel.SetSizer(scr_sizer)
            self.scrolled_panel.SetupScrolling()
            sizers.Add(self.scrolled_panel, 1, border=6, flag=wx.EXPAND)

        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_OPEN,   '画像を開く')
        file_menu.Append(wx.ID_SAVEAS, '保存')
        menu_bar.Append(file_menu, 'ファイル')
        self.SetMenuBar(menu_bar)
        self.Bind(wx.EVT_MENU, self._on_menu)
        acc_table = wx.AcceleratorTable([
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('O'), wx.ID_OPEN  ),
            wx.AcceleratorEntry(wx.ACCEL_CTRL, ord('S'), wx.ID_SAVEAS)
        ])
        self.SetAcceleratorTable(acc_table)

        self.CreateStatusBar()
    
        frame = self
        class _ImageFileDropTarget(wx.FileDropTarget):
            def OnDropFiles(self, x, y, filenames:list[str]):
                frame.open_image_file(filenames[0])
                return True
        self.SetDropTarget(_ImageFileDropTarget())

        self.SetSize(wx.Size(600, 900))
        self.set_status('idle')

        async def _check_api_status():
            try:
                await self.stable_diffusion.get_generation_progress()
            except:
                wx.CallAfter(lambda: self.SetStatusText('APIにアクセスできません'))
        asyncio.run_coroutine_threadsafe(_check_api_status(), self.stable_diffusion.event_loop)
        asyncio.run_coroutine_threadsafe(self.sd_options.fill_in_combo_box_choices(), self.stable_diffusion.event_loop)


    def set_image(self, image:Optional[Image.Image], snap_dir:Optional[Direction]=None):
        if image is not self.image or image is None:
            if image is not None:
                self.image = image
                self.bitmap = wx.ImageFromBuffer(image.width, image.height, image.tobytes()).ConvertToBitmap()
                self.canvas.Size = image.size # type: ignore
                self.canvas.MinSize = self.canvas.Size # type: ignore
                self.scrolled_panel.SetupScrolling()
                match snap_dir:
                    case Direction.RIGHT:
                        print(image.width - self.scrolled_panel.Size[0]) # type: ignore
                        wx.CallAfter(lambda: self.scrolled_panel.Scroll(0x7FFFFFFF, 0))
                    case Direction.DOWN:
                        wx.CallAfter(lambda: self.scrolled_panel.Scroll(0, 0x7FFFFFFF))
            else:
                self.image = None
                self.bitmap = None
            self.generate_cancel_button.Enabled = image is not None
            self.set_status()
            self.Refresh()



    def _do_draw_image(self, *_):
        if self.bitmap is not None:
            with wx.PaintDC(self.canvas) as _dc:
                dc: wx.PaintDC = _dc
                dc.DrawBitmap(self.bitmap, 0, 0)

    @property
    def selected_direction(self) -> Direction:
        for button, dir in self.direction_buttons:
            if button.Value:
                return dir
        return Direction.RIGHT

    def set_status(self, status:Optional[Status]=None, status_text:Optional[str]=None):
        if status is not None:
            self.status = status
        btn_color = wx.BLACK
        btn_enabled = self.image is not None
        match self.status:
            case 'idle':
                btn_text = '生成'
            case 'generating':
                btn_text = '中止'
                btn_color = wx.RED
            case 'cancelling':
                btn_text = '中断中...'
                btn_color = wx.RED
                btn_enabled = False
        def _update_button():
            self.generate_cancel_button.Label = btn_text
            self.generate_cancel_button.SetForegroundColour(btn_color)
            self.generate_cancel_button.Enabled = btn_enabled
            if self.status == 'idle':
                self.progress_bar.SetValue(0)
            self.consecutive_gen_button.Enabled = btn_enabled and (self.status == 'idle')
        wx.CallAfter(_update_button)
        if status_text is not None:
            wx.CallAfter(lambda: self.SetStatusText(status_text))

    def _on_menu(self, e:wx.MenuEvent):
        match e.GetId():
            case wx.ID_OPEN:
                try:
                    path = wx.FileSelector('画像を開く', default_extension='png', wildcard='PNG (*.png)|*.png')
                    if path:
                        self.open_image_file(path)
                except:
                    traceback.print_exc()
                    wx.MessageDialog(self, '画像を開けませんでした', style=wx.ICON_ERROR).ShowModal()
            case wx.ID_SAVEAS:
                if self.image is not None:
                    try:
                        path = wx.FileSelector('画像を保存', default_extension='png', wildcard='PNG (*.png)|*.png')
                        if path:
                            self.image.save(path)
                    except:
                        traceback.print_exc()
                        wx.MessageDialog(self, '画像の保存に失敗しました', style=wx.ICON_ERROR).ShowModal()
                else:
                    wx.MessageDialog(self, '保存できる画像がありません', style=wx.ICON_WARNING).ShowModal()

    def _restrict_direction_buttons(self):
        for radiobutton, dir in self.direction_buttons:
            radiobutton.Enabled = self.is_horizontal is None or (dir.is_horizontal == self.is_horizontal)

    async def _generate_coroutine(self, n_consecutive:Optional[int]=None):
        try:
            if self.status == 'idle':
                try:
                    assert self.image

                    direction = self.selected_direction

                    # 生成方向と垂直方向の大きさをあらかじめ設定してある大きさにリサイズする
                    if direction.is_horizontal:
                        if self.image.height != IMAGE_SIZE:
                            new_size = (round(self.image.width / self.image.height * IMAGE_SIZE), IMAGE_SIZE)
                            self.set_image(self.image.resize(new_size), direction)
                    else:
                        if self.image.width != IMAGE_SIZE:
                            new_size = (IMAGE_SIZE, round(self.image.height / self.image.width * IMAGE_SIZE))
                            self.set_image(self.image.resize(new_size), direction)
                    #self.is_horizontal = direction.is_horizontal
                    #wx.CallAfter(self._restrict_direction_buttons)

                    self.set_status('generating', '生成中...')
                    async def update_progress_bar():
                        while self.status == 'generating':
                            progress = await self.stable_diffusion.get_generation_progress()
                            if progress is not None:
                                wx.CallAfter(lambda: self.progress_bar.SetValue(int(progress * 100))) # type: ignore
                            await asyncio.sleep(0.5)
                    asyncio.create_task(update_progress_bar())

                    if n_consecutive is None:
                        # 1回生成
                        result = await self.stable_diffusion.expand_generatively(self.image, self.gen_width_control.GetValue(), direction, IMAGE_SIZE, self.sd_options.to_dict())
                    else:
                        # 連続生成
                        result = self.image
                        for iteration in range(n_consecutive):
                            self.set_status(None, f'生成中 ({(iteration + 1)}/{n_consecutive})')
                            result = await self.stable_diffusion.expand_generatively(result, self.gen_width_control.GetValue(), direction, IMAGE_SIZE, self.sd_options.to_dict())
                            if self.status == 'cancelling':
                                break
                            wx.CallAfter(lambda: self.set_image(result, direction))

                    if self.status != 'cancelling':
                        wx.CallAfter(lambda: self.set_image(result, direction))
                        self.set_status('idle', '生成が完了しました')
                    else:
                        self.set_status('idle', '生成を中断しました')
                except:
                    self.set_status('idle', '生成がエラーで停止しました')
                    raise
            else:
                self.set_status('cancelling', '中断しています...')
                await self.stable_diffusion.interrupt_generation()
        except:
            traceback.print_exc()

    def open_image_file(self, path:str):
        img:Image.Image = Image.open(path)
        self.set_image(img)
        self.is_horizontal = None


def main():
    stable_diffusion = StableDiffusion()

    app = wx.App()

    main_frame = MainFrame(stable_diffusion)
    main_frame.Show()

    app.MainLoop()


if __name__ == '__main__':
    main()
