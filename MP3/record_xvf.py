"""
XVF3800 麦克风录音工具
用于录制 WAV 文件进行唤醒词测试

使用方法：
1. 运行脚本：python MP3/record_xvf.py
2. 按下 1 开始录制
3. 按下 2 结束录制
4. 按下 3 弹出文件管理器选择保存地址
5. 按下 q 退出程序
"""

import pyaudio
import wave
import threading
import time
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# 添加项目路径（确保可以导入 wakefusion）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wakefusion.drivers.audio_driver import XVF3800Driver, DeviceInfo


class XVFRecorder:
    """XVF3800 录音器"""
    
    def __init__(self):
        self.pyaudio = None
        self.stream = None
        self.device_info: DeviceInfo = None
        
        # 录音参数
        self.sample_rate = 16000  # 16kHz，匹配 KWS 模型要求
        self.channels = 1  # 单声道
        self.chunk = 1024  # 每次读取的帧数
        self.format = pyaudio.paInt16  # 16-bit PCM
        
        # 录音状态
        self.is_recording = False
        self.audio_frames = []
        self.recording_thread = None
        
        # 批量录制相关
        self.mode = "manual"  # "manual" 或 "auto"
        self.save_dir = None  # 保存目录
        self.file_prefix = "recording"  # 文件前缀
        self.file_counter = 1  # 文件编号计数器
        self.total_recorded = 0  # 总录制数量
        
        # 自动模式参数
        self.auto_count = 10  # 自动录制数量
        self.auto_duration = 2.0  # 每段录制时长（秒）
        self.auto_interval = 3.0  # 录制间隔（秒）
        self.auto_current = 0  # 当前自动录制编号
        self.auto_timer_thread = None
        self.auto_running = False
        
        # 初始化 PyAudio
        self.pyaudio = pyaudio.PyAudio()
        
    def find_xvf_device(self) -> bool:
        """查找 XVF3800 设备"""
        print("\n" + "=" * 60)
        print("正在查找 XVF3800 设备...")
        print("=" * 60)
        
        driver = XVF3800Driver(device_match="XVF3800", sample_rate=self.sample_rate)
        self.device_info = driver.find_device()
        
        if self.device_info and self.device_info.is_xvf3800:
            print(f"\n✅ 找到 XVF3800 设备:")
            print(f"   设备名称: {self.device_info.name}")
            print(f"   设备索引: {self.device_info.index}")
            print(f"   采样率: {self.device_info.sample_rate} Hz")
            print(f"   声道数: {self.device_info.channels}")
            return True
        else:
            print("\n❌ 未找到 XVF3800 设备")
            print("   请检查:")
            print("   1. XVF3800 是否已通过 USB 连接")
            print("   2. Windows 声音设置中设备是否可见")
            print("   3. 设备是否已启用")
            return False
    
    def start_recording(self):
        """开始录音（手动模式）"""
        if self.mode != "manual":
            print("\n⚠️  当前不是手动模式，请先切换到手动模式（按 m）")
            return
        
        if self.is_recording:
            print("\n⚠️  已经在录音中，请先停止当前录音")
            return
        
        if not self.device_info:
            print("\n❌ 未找到音频设备，无法开始录音")
            return
        
        print(f"\n{'='*60}")
        print(f"🎤 [手动模式] 开始录音...")
        print(f"{'='*60}")
        print(f"   (按下 2 停止录音)")
        
        self.is_recording = True
        self.audio_frames = []
        
        try:
            # 打开音频流
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_info.index,
                frames_per_buffer=self.chunk
            )
            
            # 启动录音线程
            self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.recording_thread.start()
            
            print("   ✅ 录音已开始")
            
        except Exception as e:
            print(f"\n❌ 启动录音失败: {e}")
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
    
    def _record_audio(self):
        """录音线程函数"""
        try:
            while self.is_recording:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.audio_frames.append(data)
        except Exception as e:
            print(f"\n❌ 录音过程中出错: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """停止录音（手动模式）"""
        if not self.is_recording:
            print("\n⚠️  当前没有在录音")
            return
        
        print(f"\n{'='*60}")
        print(f"⏹️  [手动模式] 停止录音...")
        print(f"{'='*60}")
        self.is_recording = False
        
        # 等待录音线程结束
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # 计算录音时长
        duration = len(self.audio_frames) * self.chunk / self.sample_rate
        print(f"   ✅ 录音已停止")
        print(f"   📊 录音时长: {duration:.2f} 秒")
        print(f"   📦 音频帧数: {len(self.audio_frames)}")
        
        if len(self.audio_frames) == 0:
            print("   ⚠️  警告: 没有录制到任何音频数据")
        
        # 批量录制模式：自动保存
        if self.save_dir:
            print(f"   💾 自动保存中...")
            self.save_recording(auto_save=True)
            print(f"   ✅ 已自动保存（总录制: {self.total_recorded} 段）")
    
    def select_save_directory(self):
        """选择保存目录（批量录制时只需选择一次）"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        default_dir = str(Path(__file__).parent)
        
        save_dir = filedialog.askdirectory(
            title="选择批量录制保存目录",
            initialdir=default_dir
        )
        
        root.destroy()
        
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✅ 保存目录已设置: {self.save_dir}")
            return True
        else:
            print("\n❌ 未选择保存目录")
            return False
    
    def save_recording(self, auto_save=False):
        """保存录音文件"""
        if not self.audio_frames or len(self.audio_frames) == 0:
            if not auto_save:
                print("\n⚠️  没有可保存的录音数据")
                print("   请先进行录音（按下 1 开始，按下 2 停止）")
            return None
        
        # 批量录制模式：自动保存到指定目录
        if self.mode == "auto" or (self.save_dir and auto_save):
            if not self.save_dir:
                if not self.select_save_directory():
                    return None
            
            # 自动生成文件名
            filename = f"{self.file_prefix}_{self.file_counter:04d}.wav"
            file_path = self.save_dir / filename
            self.file_counter += 1
        else:
            # 手动模式：弹出文件保存对话框
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            default_dir = str(self.save_dir) if self.save_dir else str(Path(__file__).parent)
            default_filename = f"{self.file_prefix}_{int(time.time())}.wav"
            
            file_path = filedialog.asksaveasfilename(
                title="保存录音文件",
                initialdir=default_dir,
                initialfile=default_filename,
                defaultextension=".wav",
                filetypes=[
                    ("WAV 音频文件", "*.wav"),
                    ("所有文件", "*.*")
                ]
            )
            
            root.destroy()
            
            if not file_path:
                if not auto_save:
                    print("\n❌ 未选择保存路径，取消保存")
                return None
            
            file_path = Path(file_path)
        
        try:
            # 保存为 WAV 文件
            with wave.open(str(file_path), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            file_size = file_path.stat().st_size / 1024  # KB
            duration = len(self.audio_frames) * self.chunk / self.sample_rate
            self.total_recorded += 1
            
            print(f"\n✅ 录音已保存 [{self.total_recorded}]:")
            print(f"   文件路径: {file_path}")
            print(f"   文件大小: {file_size:.2f} KB")
            print(f"   录音时长: {duration:.2f} 秒")
            print(f"   采样率: {self.sample_rate} Hz")
            print(f"   声道数: {self.channels}")
            
            # 清空当前录音数据（批量录制时）
            if auto_save:
                self.audio_frames = []
            
            return file_path
            
        except Exception as e:
            print(f"\n❌ 保存文件失败: {e}")
            return None
    
    def set_auto_params(self, count=None, duration=None, interval=None, prefix=None):
        """设置自动录制参数"""
        if count is not None:
            self.auto_count = max(1, int(count))
        if duration is not None:
            self.auto_duration = max(0.5, float(duration))
        if interval is not None:
            self.auto_interval = max(0.5, float(interval))
        if prefix is not None:
            self.file_prefix = prefix
        
        print(f"\n📋 自动录制参数已设置:")
        print(f"   录制数量: {self.auto_count}")
        print(f"   每段时长: {self.auto_duration:.1f} 秒")
        print(f"   录制间隔: {self.auto_interval:.1f} 秒")
        print(f"   文件前缀: {self.file_prefix}")
    
    def start_auto_recording(self):
        """开始自动批量录制"""
        if self.mode != "auto":
            print("\n⚠️  当前不是自动模式，请先切换到自动模式（按 m）")
            return
        
        if self.auto_running:
            print("\n⚠️  自动录制已在运行中")
            return
        
        if not self.save_dir:
            print("\n📁 请先选择保存目录...")
            if not self.select_save_directory():
                return
        
        print(f"\n🚀 开始自动批量录制...")
        print(f"   模式: 自动模式")
        print(f"   总数量: {self.auto_count} 段")
        print(f"   每段时长: {self.auto_duration:.1f} 秒")
        print(f"   录制间隔: {self.auto_interval:.1f} 秒")
        print(f"   保存目录: {self.save_dir}")
        print(f"   文件前缀: {self.file_prefix}")
        print()
        
        self.auto_current = 0
        self.auto_running = True
        self.file_counter = 1
        
        # 启动自动录制线程
        self.auto_timer_thread = threading.Thread(target=self._auto_record_loop, daemon=True)
        self.auto_timer_thread.start()
    
    def _auto_record_loop(self):
        """自动录制循环"""
        try:
            for i in range(self.auto_count):
                if not self.auto_running:
                    break
                
                self.auto_current = i + 1
                
                # 显示开始录制
                print(f"\n{'='*60}")
                print(f"🎤 [自动模式] 开始录制第 {self.auto_current}/{self.auto_count} 段")
                print(f"{'='*60}")
                
                # 开始录制
                self.is_recording = True
                self.audio_frames = []
                
                try:
                    if not self.stream:
                        self.stream = self.pyaudio.open(
                            format=self.format,
                            channels=self.channels,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=self.device_info.index,
                            frames_per_buffer=self.chunk
                        )
                    
                    # 录制指定时长
                    start_time = time.time()
                    while self.is_recording and (time.time() - start_time) < self.auto_duration:
                        data = self.stream.read(self.chunk, exception_on_overflow=False)
                        self.audio_frames.append(data)
                        
                        # 实时显示录制进度
                        elapsed = time.time() - start_time
                        remaining = max(0, self.auto_duration - elapsed)
                        print(f"\r   ⏺️  录制中... {elapsed:.1f}s / {self.auto_duration:.1f}s (剩余 {remaining:.1f}s)", end='', flush=True)
                    
                    # 停止录制
                    self.is_recording = False
                    print()  # 换行
                    
                    # 保存录音
                    print(f"   💾 正在保存第 {self.auto_current} 段...")
                    file_path = self.save_recording(auto_save=True)
                    
                    if file_path:
                        print(f"   ✅ 第 {self.auto_current}/{self.auto_count} 段已保存")
                    else:
                        print(f"   ❌ 第 {self.auto_current} 段保存失败")
                    
                except Exception as e:
                    print(f"\n   ❌ 录制第 {self.auto_current} 段时出错: {e}")
                    self.is_recording = False
                
                # 如果不是最后一段，等待间隔
                if i < self.auto_count - 1 and self.auto_running:
                    print(f"\n   ⏳ 等待 {self.auto_interval:.1f} 秒后录制下一段...")
                    for wait_time in range(int(self.auto_interval)):
                        if not self.auto_running:
                            break
                        remaining = self.auto_interval - wait_time
                        print(f"\r   ⏳ 等待中... {remaining:.1f}s", end='', flush=True)
                        time.sleep(1)
                    print()  # 换行
            
            # 自动录制完成
            self.auto_running = False
            print(f"\n{'='*60}")
            print(f"✅ 自动批量录制完成！")
            print(f"   总录制: {self.auto_current} 段")
            print(f"   保存目录: {self.save_dir}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n❌ 自动录制过程出错: {e}")
            self.auto_running = False
            self.is_recording = False
    
    def stop_auto_recording(self):
        """停止自动录制"""
        if not self.auto_running:
            print("\n⚠️  当前没有在自动录制")
            return
        
        print("\n⏹️  正在停止自动录制...")
        self.auto_running = False
        self.is_recording = False
        
        # 等待线程结束
        if self.auto_timer_thread and self.auto_timer_thread.is_alive():
            self.auto_timer_thread.join(timeout=3.0)
        
        print("   ✅ 自动录制已停止")
    
    def switch_mode(self):
        """切换录制模式"""
        if self.mode == "manual":
            self.mode = "auto"
            print("\n🔄 已切换到: 自动模式")
            print("   提示: 使用 'p' 设置参数，使用 'a' 开始自动录制")
        else:
            self.mode = "manual"
            if self.auto_running:
                self.stop_auto_recording()
            print("\n🔄 已切换到: 手动模式")
            print("   提示: 使用 '1' 开始，'2' 停止，'3' 保存")
    
    def cleanup(self):
        """清理资源"""
        # 停止自动录制
        if self.auto_running:
            self.stop_auto_recording()
        
        if self.is_recording:
            self.stop_recording()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None


def get_user_input():
    """非阻塞获取用户输入"""
    import msvcrt  # Windows 专用
    
    if msvcrt.kbhit():
        key = msvcrt.getch().decode('utf-8').lower()
        return key
    return None


def main():
    """主函数"""
    print("=" * 60)
    print("🎤 XVF3800 麦克风录音工具")
    print("=" * 60)
    
    recorder = XVFRecorder()
    
    try:
        # 查找设备
        if not recorder.find_xvf_device():
            print("\n按任意键退出...")
            input()
            return
        
        # 显示操作说明
        def print_status():
            """打印当前状态和操作说明"""
            print("\n" + "=" * 60)
            print(f"📊 当前状态")
            print("=" * 60)
            mode_text = "🔄 自动模式" if recorder.mode == "auto" else "✋ 手动模式"
            print(f"   模式: {mode_text}")
            print(f"   总录制: {recorder.total_recorded} 段")
            if recorder.save_dir:
                print(f"   保存目录: {recorder.save_dir}")
            print("=" * 60)
            
            if recorder.mode == "manual":
                print("📋 手动模式操作:")
                print("   按下 1: 开始录音")
                print("   按下 2: 停止录音（如果设置了保存目录，会自动保存）")
                print("   按下 3: 保存录音文件（弹出文件对话框）")
                print("   按下 d: 设置保存目录（批量录制时只需设置一次）")
            else:
                print("📋 自动模式操作:")
                print("   按下 p: 设置自动录制参数（数量、时长、间隔、前缀）")
                print("   按下 d: 设置保存目录（必须先设置）")
                print("   按下 a: 开始自动批量录制")
                print("   按下 s: 停止自动录制")
            
            print("   按下 m: 切换模式（手动 ↔ 自动）")
            print("   按下 q: 退出程序")
            print("=" * 60)
            if recorder.is_recording:
                print("   ⏺️  状态: 正在录音中...")
            elif recorder.auto_running:
                print(f"   ⏺️  状态: 自动录制中... ({recorder.auto_current}/{recorder.auto_count})")
            else:
                print("   ⏸️  状态: 等待操作...")
            print()
        
        print_status()
        
        # 主循环
        while True:
            key = get_user_input()
            
            if key == '1':
                recorder.start_recording()
            elif key == '2':
                recorder.stop_recording()
            elif key == '3':
                recorder.save_recording()
            elif key == 'm':
                recorder.switch_mode()
                print_status()
            elif key == 'd':
                recorder.select_save_directory()
            elif key == 'p':
                # 设置自动录制参数
                if recorder.mode == "auto":
                    print("\n📋 设置自动录制参数:")
                    try:
                        count = input(f"   录制数量 (当前: {recorder.auto_count}): ").strip()
                        duration = input(f"   每段时长/秒 (当前: {recorder.auto_duration:.1f}): ").strip()
                        interval = input(f"   录制间隔/秒 (当前: {recorder.auto_interval:.1f}): ").strip()
                        prefix = input(f"   文件前缀 (当前: {recorder.file_prefix}): ").strip()
                        
                        recorder.set_auto_params(
                            count=int(count) if count else None,
                            duration=float(duration) if duration else None,
                            interval=float(interval) if interval else None,
                            prefix=prefix if prefix else None
                        )
                    except ValueError as e:
                        print(f"   ❌ 参数输入错误: {e}")
                else:
                    print("\n⚠️  请先切换到自动模式（按 m）")
            elif key == 'a':
                recorder.start_auto_recording()
            elif key == 's':
                recorder.stop_auto_recording()
            elif key == 'q':
                print("\n👋 退出程序...")
                break
            
            time.sleep(0.1)  # 避免 CPU 占用过高
    
    except KeyboardInterrupt:
        print("\n\n👋 程序被中断，正在退出...")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recorder.cleanup()
        print("✅ 资源已清理，程序退出")


if __name__ == "__main__":
    # Windows 下使用 msvcrt 进行非阻塞输入
    try:
        import msvcrt
    except ImportError:
        print("❌ 错误: 此脚本仅支持 Windows 系统")
        print("   在 Linux/Mac 上，请使用其他方式获取键盘输入")
        sys.exit(1)
    
    main()
