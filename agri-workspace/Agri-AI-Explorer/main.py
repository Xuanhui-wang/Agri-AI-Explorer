import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import convolve2d

# Set appearance mode and color theme
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")

# Configure Matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] # Set default font
plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("“农智探客”(Agri-AI Explorer) - 农业AI算法可视化平台")
        self.geometry("1100x750")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="农智探客\nAgri-AI Explorer", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, text="1. 特征透视：病斑提取", command=self.show_cnn_lab)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame, text="2. 回归分析：产量预测", command=self.show_data_lab)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_3 = ctk.CTkButton(self.sidebar_frame, text="3. 训练可视：模型调参", command=self.show_model_lab)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_4 = ctk.CTkButton(self.sidebar_frame, text="4. 智能答疑：助教中心", command=self.show_ai_assistant)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_button_5 = ctk.CTkButton(self.sidebar_frame, text="5. 交互编程：农智沙箱", command=self.show_sandbox_lab, fg_color="#2E8B57", hover_color="#006400")
        self.sidebar_button_5.grid(row=5, column=0, padx=20, pady=10)

        self.sidebar_button_6 = ctk.CTkButton(self.sidebar_frame, text="6. 仿真实验：农机驾驶", command=self.show_simulation_lab, fg_color="#D2691E", hover_color="#A0522D")
        self.sidebar_button_6.grid(row=6, column=0, padx=20, pady=10)

        # Main frame content
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Default to CNN Lab to show the new feature directly
        self.current_frame = None
        self.show_cnn_lab()

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_cnn_lab(self):
        self.clear_main_frame()
        title = ctk.CTkLabel(self.main_frame, text="卷积神经网络(CNN)底层原理：小麦病害特征提取透视", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        desc = ctk.CTkLabel(self.main_frame, text="【击破痛点】抛弃抽象数学推导！本模块使用真实的《跨品种小麦赤霉病(FHB)无人机影像数据集》，\n包含豫农912、郑麦0923、郑麦1860、郑麦9134四个品种的轻度发病样本。\n通过切换不同品种的真实图像和卷积核，直观理解CNN如何通过矩阵计算提取真实农业场景中的病斑特征。", justify="left", font=ctk.CTkFont(size=14))
        desc.grid(row=1, column=0, padx=20, pady=5, sticky="nw")

        controls_frame = ctk.CTkFrame(self.main_frame)
        controls_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(controls_frame, text="发病程度 (优先级1):").grid(row=0, column=0, padx=10, pady=10)
        
        self.severity_var = ctk.StringVar(value="中度 (Moderate)")
        self.severity_combo = ctk.CTkComboBox(controls_frame, values=[
            "健康 (Healthy)",
            "轻度 (Mild)", 
            "中度 (Moderate)", 
            "重度 (Severe)"
        ], variable=self.severity_var, width=120, command=self.load_and_apply_convolution)
        self.severity_combo.grid(row=0, column=1, padx=10, pady=10)

        ctk.CTkLabel(controls_frame, text="小麦品种 (优先级2):").grid(row=0, column=2, padx=10, pady=10)
        
        self.wheat_variety_var = ctk.StringVar(value="豫农912 (中感)")
        self.variety_combo = ctk.CTkComboBox(controls_frame, values=[
            "豫农912 (中感)", 
            "郑麦0923 (中感)", 
            "郑麦1860 (高感)", 
            "郑麦9134 (高感)"
        ], variable=self.wheat_variety_var, width=150, command=self.load_and_apply_convolution)
        self.variety_combo.grid(row=0, column=3, padx=10, pady=10)

        ctk.CTkLabel(controls_frame, text="选择卷积核(Kernel):").grid(row=0, column=4, padx=10, pady=10)
        
        self.kernel_var = ctk.StringVar(value="边缘检测 (提取病斑边缘)")
        self.kernel_combo = ctk.CTkComboBox(controls_frame, values=["边缘检测 (提取病斑边缘)", "锐化处理 (突出病斑细节)", "平滑滤波 (消除背景噪点)", "浮雕效果 (立体化纹理)"], variable=self.kernel_var, width=180, command=self.apply_convolution)
        self.kernel_combo.grid(row=0, column=5, padx=10, pady=10)

        self.anim_btn = ctk.CTkButton(controls_frame, text="▶ 播放矩阵计算动画", command=self.play_animation, fg_color="#CD5C5C", hover_color="#8B0000")
        self.anim_btn.grid(row=0, column=6, padx=10, pady=10)

        self.fig_cnn, self.axs_cnn = plt.subplots(1, 3, figsize=(11, 4.5))
        self.fig_cnn.suptitle("CNN 卷积层单步计算可视化 (真实图像灰度矩阵 * 卷积核矩阵)", fontsize=18, color="black", fontweight="bold")
        self.fig_cnn.patch.set_facecolor('white')
        for ax in self.axs_cnn:
            ax.set_facecolor('white')
            ax.tick_params(colors='black', labelsize=10)
            ax.title.set_color('black')
            ax.title.set_fontsize(14)
            ax.title.set_fontweight('bold')

        self.canvas_cnn = FigureCanvasTkAgg(self.fig_cnn, master=self.main_frame)
        self.canvas_cnn.get_tk_widget().grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(3, weight=1)

        self.load_and_apply_convolution()

    def load_and_apply_convolution(self, *args):
        import os
        from PIL import Image, ImageOps
        variety_mapping = {
            "豫农912 (中感)": "yunong912",
            "郑麦0923 (中感)": "zhengmai0923",
            "郑麦1860 (高感)": "zhengmai1860",
            "郑麦9134 (高感)": "zhengmai9134"
        }
        
        severity_mapping = {
            "健康 (Healthy)": "_healthy",
            "轻度 (Mild)": "",
            "中度 (Moderate)": "_mod",
            "重度 (Severe)": "_sev"
        }
        
        base_name = variety_mapping.get(self.wheat_variety_var.get())
        suffix = severity_mapping.get(self.severity_var.get())
        img_name = f"{base_name}{suffix}.png"
        
        img_path = os.path.join(os.path.dirname(__file__), "assets", img_name)
        
        try:
            color_img = Image.open(img_path).convert('RGB')
            color_img = color_img.resize((150, 150))
            self.wheat_img_color = np.array(color_img) / 255.0
            
            img_gray = color_img.convert('L')
            self.wheat_img = np.array(img_gray) / 255.0
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            self.wheat_img_color, self.wheat_img = self.generate_wheat_leaf() # fallback
            
        self.apply_convolution()

    def generate_wheat_leaf(self):
        # 模拟生成一个带有病斑的小麦叶片灰度图与彩色图
        x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        # 叶片形状
        leaf = np.exp(-1.5*(x**2 + 5*y**2))
        # 背景噪点
        noise = np.random.randn(100, 100) * 0.05
        
        is_healthy = "健康" in self.severity_var.get()
        if is_healthy:
            img_gray = leaf + noise
            # 伪彩色(绿色)
            color_img = np.zeros((100, 100, 3))
            color_img[:, :, 1] = leaf + noise * 0.5 
        else:
            # 加入几个小麦病斑点
            spot1 = np.exp(-100 * ((x - 0.2)**2 + (y - 0.1)**2))
            spot2 = np.exp(-150 * ((x + 0.1)**2 + (y + 0.3)**2))
            img_gray = leaf - 0.7 * spot1 - 0.6 * spot2 + noise
            
            # 伪彩色病斑
            color_img = np.zeros((100, 100, 3))
            color_img[:, :, 1] = leaf + noise * 0.5 
            color_img[:, :, 0] += 0.8 * spot1 + 0.9 * spot2 
            color_img[:, :, 1] -= 0.5 * spot1 + 0.4 * spot2 
            
        img_gray = np.clip(img_gray, 0, 1)
        color_img = np.clip(color_img, 0, 1)
        return color_img, img_gray

    def play_animation(self):
        import os
        import subprocess
        import tkinter.messagebox as messagebox
        video_path = r"D:\3.创AI案例征集指南与模板\CNNWheatDisease.mp4"
        if os.path.exists(video_path):
            import sys
            if sys.platform == "win32":
                os.startfile(video_path)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, video_path])
        else:
            messagebox.showwarning("动画缺失", f"未找到动画视频文件：\n{video_path}")

    def apply_convolution(self, *args):
        kernel_name = self.kernel_var.get()
        
        if "边缘检测" in kernel_name:
            kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
        elif "锐化" in kernel_name:
            kernel = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])
        elif "平滑" in kernel_name:
            kernel = np.array([[ 1,  1,  1],
                               [ 1,  1,  1],
                               [ 1,  1,  1]]) / 9.0
        else: # 浮雕
            kernel = np.array([[-2, -1,  0],
                               [-1,  1,  1],
                               [ 0,  1,  2]])

        # Convolve
        feature_map = convolve2d(self.wheat_img, kernel, mode='valid')

        # Update plots
        self.axs_cnn[0].clear()
        self.axs_cnn[1].clear()
        self.axs_cnn[2].clear()

        for ax in self.axs_cnn:
            ax.set_facecolor('white')

        self.axs_cnn[0].set_title("输入层: 原始RGB彩色图像\n(计算时转灰度)", color='black', fontweight='bold', fontsize=14, pad=10)
        if hasattr(self, 'wheat_img_color'):
            self.axs_cnn[0].imshow(self.wheat_img_color)
        else:
            self.axs_cnn[0].imshow(self.wheat_img, cmap='gray')
        self.axs_cnn[0].set_xticks([])
        self.axs_cnn[0].set_yticks([])
        for spine in self.axs_cnn[0].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)

        self.axs_cnn[1].set_title("卷积核 (Weight)", color='black', fontweight='bold', fontsize=14, pad=10)
        self.axs_cnn[1].matshow(np.zeros_like(kernel), cmap='binary', vmin=0, vmax=1)
        for i in range(kernel.shape[0] + 1):
            self.axs_cnn[1].axhline(i - 0.5, color='red', linewidth=2)
        for j in range(kernel.shape[1] + 1):
            self.axs_cnn[1].axvline(j - 0.5, color='red', linewidth=2)
            
        for (i, j), z in np.ndenumerate(kernel):
            self.axs_cnn[1].text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color='black', fontweight='bold', fontsize=16)
        self.axs_cnn[1].set_xticks([])
        self.axs_cnn[1].set_yticks([])
        for spine in self.axs_cnn[1].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)

        self.axs_cnn[2].set_title("特征图 (Feature Map)", color='black', fontweight='bold', fontsize=14, pad=10)
        self.axs_cnn[2].imshow(feature_map, cmap='viridis')
        self.axs_cnn[2].set_xticks([])
        self.axs_cnn[2].set_yticks([])
        for spine in self.axs_cnn[2].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)

        self.fig_cnn.tight_layout(rect=[0, 0, 1, 0.92])
        self.canvas_cnn.draw()


    def show_data_lab(self):
        self.clear_main_frame()
        title = ctk.CTkLabel(self.main_frame, text="农业特色数据集实验室：水稻产量预测 (特征与调参)", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        desc = ctk.CTkLabel(self.main_frame, text="【交互探索】学生可通过勾选特征、拖动调整数据噪声和模型超参数，\n实时观察机器学习模型(如线性回归、随机森林)在回归预测任务上的表现，深刻理解过拟合、欠拟合与特征重要性。", justify="left", font=ctk.CTkFont(size=14))
        desc.grid(row=1, column=0, padx=20, pady=5, sticky="nw")

        # Layout for Data Lab
        content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        # Left panel: Data Distribution & Controls
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        
        self.fig_data, self.axs_data = plt.subplots(2, 2, figsize=(5, 3.5))
        self.fig_data.suptitle("农情特征与水稻产量的关系分布", fontsize=14, color="#2F4F4F", fontweight="bold")
        self.fig_data.patch.set_facecolor('white')
        for ax in self.axs_data.flat:
            ax.set_facecolor('#F8F8FF')
            ax.tick_params(colors='#556B2F', labelsize=9)
            ax.xaxis.label.set_color('#556B2F')
            ax.yaxis.label.set_color('#556B2F')
            ax.title.set_color('#2E8B57')
            ax.title.set_fontsize(12)
            ax.title.set_fontweight('bold')

        self.canvas_data = FigureCanvasTkAgg(self.fig_data, master=left_panel)
        self.canvas_data.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
        
        # Interactive Controls for Data
        data_ctrl = ctk.CTkFrame(left_panel, fg_color="transparent")
        data_ctrl.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(data_ctrl, text="拖拽调整数据噪声扰动量:").grid(row=0, column=0, sticky="w", padx=5)
        self.noise_slider = ctk.CTkSlider(data_ctrl, from_=0, to=1000, command=self.generate_and_plot_data)
        self.noise_slider.set(300)
        self.noise_slider.grid(row=0, column=1, sticky="ew", padx=5)

        # Right panel: Model Prediction
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.grid(row=0, column=1, padx=(10, 0), sticky="nsew")

        ctk.CTkLabel(right_panel, text="模型配置与实时评估", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 5))
        
        controls = ctk.CTkFrame(right_panel, fg_color="transparent")
        controls.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(controls, text="选择算法模型:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.reg_model_var = ctk.StringVar(value="多元线性回归 (Linear)")
        model_combo = ctk.CTkComboBox(controls, values=["多元线性回归 (Linear)", "随机森林回归 (Random Forest)"], variable=self.reg_model_var, width=190, command=self.train_and_predict_yield)
        model_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ctk.CTkLabel(controls, text="随机森林树数量(仅RF有效):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.rf_trees_slider = ctk.CTkSlider(controls, from_=1, to=150, command=self.train_and_predict_yield)
        self.rf_trees_slider.set(50)
        self.rf_trees_slider.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        feat_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        feat_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(feat_frame, text="拖拽/勾选输入特征:").pack(side="left", padx=5)
        
        self.use_rain = ctk.BooleanVar(value=True)
        self.use_temp = ctk.BooleanVar(value=True)
        self.use_light = ctk.BooleanVar(value=True)
        self.use_fert = ctk.BooleanVar(value=True)
        
        ctk.CTkCheckBox(feat_frame, text="降水", variable=self.use_rain, command=self.train_and_predict_yield).pack(side="left", padx=2)
        ctk.CTkCheckBox(feat_frame, text="温度", variable=self.use_temp, command=self.train_and_predict_yield).pack(side="left", padx=2)
        ctk.CTkCheckBox(feat_frame, text="光照", variable=self.use_light, command=self.train_and_predict_yield).pack(side="left", padx=2)
        ctk.CTkCheckBox(feat_frame, text="施肥", variable=self.use_fert, command=self.train_and_predict_yield).pack(side="left", padx=2)

        self.score_label = ctk.CTkLabel(right_panel, text="等待训练...", font=ctk.CTkFont(size=14), text_color="#2E8B57")
        self.score_label.pack(pady=5)

        self.fig_pred, self.ax_pred = plt.subplots(figsize=(5, 3))
        self.fig_pred.patch.set_facecolor('white')
        self.ax_pred.set_facecolor('#F8F8FF')
        self.ax_pred.tick_params(colors='#556B2F', labelsize=10)
        self.ax_pred.xaxis.label.set_color('#2F4F4F')
        self.ax_pred.yaxis.label.set_color('#2F4F4F')
        self.ax_pred.xaxis.label.set_fontsize(12)
        self.ax_pred.yaxis.label.set_fontsize(12)
        self.ax_pred.title.set_color('#2E8B57')
        self.ax_pred.title.set_fontsize(14)
        self.ax_pred.title.set_fontweight('bold')
        
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=right_panel)
        self.canvas_pred.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # Initialize data and train
        self.generate_and_plot_data()

    def generate_and_plot_data(self, *args):
        noise_level = self.noise_slider.get()
        np.random.seed(42)
        self.rain = np.random.normal(800, 200, 250)
        self.temp = np.random.normal(25, 5, 250)
        self.light = np.random.normal(2000, 300, 250)
        self.fertilizer = np.random.normal(150, 30, 250)
        # Yield is dependent on these features with some non-linearity and dynamic noise
        self.yield_kg = 5000 + 2*self.rain + 50*self.temp + 0.5*self.light + 10*self.fertilizer + 0.02*(self.rain*self.temp) + np.random.normal(0, noise_level, 250)

        self.axs_data[0, 0].clear()
        self.axs_data[0, 1].clear()
        self.axs_data[1, 0].clear()
        self.axs_data[1, 1].clear()

        self.axs_data[0, 0].scatter(self.rain, self.yield_kg, c='#4682B4', alpha=0.6, s=12)
        self.axs_data[0, 0].set_title('降水量 vs 产量')
        self.axs_data[0, 1].scatter(self.temp, self.yield_kg, c='#D2691E', alpha=0.6, s=12)
        self.axs_data[0, 1].set_title('温度 vs 产量')
        self.axs_data[1, 0].scatter(self.light, self.yield_kg, c='#DAA520', alpha=0.6, s=12)
        self.axs_data[1, 0].set_title('光照 vs 产量')
        self.axs_data[1, 1].scatter(self.fertilizer, self.yield_kg, c='#6B8E23', alpha=0.6, s=12)
        self.axs_data[1, 1].set_title('施肥量 vs 产量')
        
        self.fig_data.tight_layout()
        self.canvas_data.draw()
        
        self.train_and_predict_yield()

    def train_and_predict_yield(self, *args):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        # Prepare data dynamically based on checked features
        features = []
        if self.use_rain.get(): features.append(self.rain)
        if self.use_temp.get(): features.append(self.temp)
        if self.use_light.get(): features.append(self.light)
        if self.use_fert.get(): features.append(self.fertilizer)
        
        if len(features) == 0:
            self.score_label.configure(text="请至少勾选一个特征进行预测！", text_color="red")
            self.ax_pred.clear()
            self.ax_pred.text(0.5, 0.5, "缺少特征输入", ha='center', va='center', color='red', fontsize=12)
            self.canvas_pred.draw()
            return

        X = np.column_stack(features)
        y = self.yield_kg

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_name = self.reg_model_var.get()
        if "Linear" in model_name:
            model = LinearRegression()
            self.rf_trees_slider.configure(state="disabled")
        else:
            trees = int(self.rf_trees_slider.get())
            model = RandomForestRegressor(n_estimators=trees, random_state=42)
            self.rf_trees_slider.configure(state="normal")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        
        # Color coding score
        score_color = "#2E8B57" if r2 > 0.7 else ("#DAA520" if r2 > 0.4 else "red")
        
        noise = int(self.noise_slider.get())
        tree_text = f", 树={int(self.rf_trees_slider.get())}" if "Random" in model_name else ""
        self.score_label.configure(text=f"R²决定系数: {r2:.4f} (噪声:{noise}{tree_text}, 特征数:{len(features)})", text_color=score_color)

        self.ax_pred.clear()
        
        import matplotlib.ticker as ticker
        self.ax_pred.xaxis.set_major_locator(ticker.AutoLocator())
        self.ax_pred.yaxis.set_major_locator(ticker.AutoLocator())

        self.ax_pred.scatter(y_test, y_pred, color="#4682B4", alpha=0.7, edgecolors="white")
        
        # Plot y=x line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        self.ax_pred.plot([min_val, max_val], [min_val, max_val], color="#B22222", linestyle="--", linewidth=2, label="完美预测线 (y=x)")
        
        self.ax_pred.set_title(f"{model_name} 预测结果", fontsize=11)
        self.ax_pred.set_xlabel("真实产量 (kg/ha)")
        self.ax_pred.set_ylabel("预测产量 (kg/ha)")
        self.ax_pred.legend(loc="upper left")
        self.fig_pred.tight_layout()
        self.canvas_pred.draw()

    def show_model_lab(self):
        self.clear_main_frame()
        title = ctk.CTkLabel(self.main_frame, text="算法训练过程可视化：多层感知机 (MLP)", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        desc = ctk.CTkLabel(self.main_frame, text="【击破痛点】拒绝做盲目的“调参侠”。通过动态绘制损失函数(Loss)曲线，让学生直观地看见学习率(LR)大小如何影响模型收敛情况。\n亲眼目睹何为梯度下降、何为模型震荡，深刻理解超参数背后的数学原理。", justify="left", font=ctk.CTkFont(size=14))
        desc.grid(row=1, column=0, padx=20, pady=10, sticky="nw")

        controls_frame = ctk.CTkFrame(self.main_frame)
        controls_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        ctk.CTkLabel(controls_frame, text="学习率 (Learning Rate):").grid(row=0, column=0, padx=10, pady=10)
        self.lr_slider = ctk.CTkSlider(controls_frame, from_=0.001, to=0.1, number_of_steps=100)
        self.lr_slider.grid(row=0, column=1, padx=10, pady=10)
        self.lr_slider.set(0.01)

        ctk.CTkLabel(controls_frame, text="训练轮数 (Epochs):").grid(row=0, column=2, padx=10, pady=10)
        self.epochs_slider = ctk.CTkSlider(controls_frame, from_=10, to=500, number_of_steps=50)
        self.epochs_slider.grid(row=0, column=3, padx=10, pady=10)
        self.epochs_slider.set(200)

        self.train_btn = ctk.CTkButton(controls_frame, text="开始动态训练", command=self.start_training)
        self.train_btn.grid(row=0, column=4, padx=20, pady=10)

        self.fig_train, self.ax_train = plt.subplots(figsize=(8, 4))
        self.fig_train.patch.set_facecolor('white')
        self.ax_train.set_facecolor('#F8F8FF')
        self.ax_train.tick_params(colors='#556B2F', labelsize=10)
        self.ax_train.xaxis.label.set_color('#2F4F4F')
        self.ax_train.yaxis.label.set_color('#2F4F4F')
        self.ax_train.xaxis.label.set_fontsize(14)
        self.ax_train.yaxis.label.set_fontsize(14)
        self.ax_train.title.set_color('#2E8B57')
        self.ax_train.title.set_fontsize(16)
        self.ax_train.title.set_fontweight('bold')
        self.ax_train.set_title("Loss 曲线 (损失函数下降图)")
        self.ax_train.set_xlabel("Epoch")
        self.ax_train.set_ylabel("Loss")

        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=self.main_frame)
        self.canvas_train.draw()
        self.canvas_train.get_tk_widget().grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(3, weight=1)

    def start_training(self):
        self.train_btn.configure(state="disabled", text="训练中...")
        lr = self.lr_slider.get()
        epochs = int(self.epochs_slider.get())

        # Generate data
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = 3*X[:, 0] + 5*X[:, 1] - 2*X[:, 2] + X[:, 3] + np.random.randn(200)*0.1
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.ax_train.clear()
        self.ax_train.set_title(f"Loss 曲线 (LR={lr:.4f}, Epochs={epochs})")
        self.ax_train.set_xlabel("Epoch")
        self.ax_train.set_ylabel("Loss")
        self.ax_train.set_xlim(0, epochs)
        self.ax_train.set_ylim(0, 10)
        self.line, = self.ax_train.plot([], [], color='#4682B4', lw=2)
        self.canvas_train.draw()

        # Run training in thread to not freeze UI
        def train_loop():
            model = MLPRegressor(hidden_layer_sizes=(16, 8), learning_rate_init=lr, max_iter=1, warm_start=True, solver='adam')
            losses = []
            for i in range(epochs):
                model.fit(X_scaled, y)
                losses.append(model.loss_)
                
                if i % max(1, epochs//50) == 0:
                    self.line.set_data(range(len(losses)), losses)
                    self.ax_train.set_ylim(0, max(losses) * 1.1)
                    self.canvas_train.draw()
                    time.sleep(0.05) # Add small delay for visualization effect

            self.line.set_data(range(len(losses)), losses)
            self.canvas_train.draw()
            self.train_btn.configure(state="normal", text="重新训练")

        threading.Thread(target=train_loop, daemon=True).start()

    def show_ai_assistant(self):
        self.clear_main_frame()
        title = ctk.CTkLabel(self.main_frame, text="AI算法助教：农业AI专属排忧解难", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.chat_box = ctk.CTkTextbox(self.main_frame, font=ctk.CTkFont(size=14))
        self.chat_box.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.chat_box.insert("0.0", "🤖 农智AI助教：你好！在搭建CNN识别小麦病害模型，或是调参过程中遇到问题了吗？\n比如：我不懂CNN卷积核里的数字代表什么？或者我该用什么网络结构？可以问我！\n\n")
        self.chat_box.configure(state="disabled")

        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text="例如：CNN的卷积核里那些-1和8是干什么的？为什么能提取病斑边缘？")
        self.chat_input.grid(row=0, column=0, padx=(10, 10), pady=10, sticky="ew")
        self.chat_input.bind("<Return>", lambda event: self.send_message())

        send_btn = ctk.CTkButton(input_frame, text="发送", width=80, command=self.send_message)
        send_btn.grid(row=0, column=1, padx=10, pady=10)

        self.main_frame.grid_rowconfigure(1, weight=1)

    def send_message(self):
        msg = self.chat_input.get()
        if not msg.strip():
            return
            
        self.chat_input.delete(0, 'end')
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"👤 你：{msg}\n\n")
        self.chat_box.insert("end", "🤖 农智AI助教：[正在思考中...]\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")
        
        # 核心：启动多线程请求大模型，防止主界面卡死
        threading.Thread(target=self.call_llm_api, args=(msg,), daemon=True).start()

    def call_llm_api(self, user_msg):
        import requests
        import json
        
        # 【配置您的国产大模型 API_KEY】
        # 这里以 DeepSeek (深度求索) 为例，它完全兼容 OpenAI 接口，目前国内热度极高
        # 您也可以换成 智谱GLM、Kimi、通义千问等，只需更改 url 和 model 即可
        API_KEY = "sk-638707bb84d84e50bb0dd71150e2bdce" 
        
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        # System prompt: 设定AI助教的角色与专业领域，保障回答质量
        system_prompt = (
            "你是一个名为'农智AI助教'的专家，负责解答学生关于农业AI、卷积神经网络(CNN)、多层感知机(MLP)和模型调参的疑问。"
            "你的回答需要：1. 专业准确 2. 结合农业场景 3. 语言生动易懂，像是老师在指导学生。字数控制在200字以内。"
        )
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.7
        }
        
        try:
            if API_KEY == "your-api-key-here":
                # 没有配置API KEY时的演示反馈
                time.sleep(1.5) # 模拟网络延迟
                ai_reply = (
                    "💡 【系统提示】当前处于本地演示模式。\n"
                    "系统已经为您搭建好了接入【国产大模型】的多线程底层代码。\n\n"
                    f"针对您的问题“{user_msg}”，\n"
                    "请在 main.py 的 call_llm_api 函数中填入您申请的 API_KEY (如 DeepSeek, 通义千问, 智谱清言等)，"
                    "即可立刻开启真实的智能对话！"
                )
            else:
                # 真实的 API 请求
                response = requests.post(url, headers=headers, json=data, timeout=15)
                response.raise_for_status()
                ai_reply = response.json()['choices'][0]['message']['content']
        except Exception as e:
            ai_reply = f"❌ 连接大模型时出现错误：\n{str(e)}\n请检查网络或 API_KEY 配置。"

        # 使用 after 回到主线程更新 UI，防止跨线程操作GUI报错
        self.after(0, self.update_chat_box, ai_reply)

    def update_chat_box(self, reply):
        self.chat_box.configure(state="normal")
        
        # 替换掉"正在思考中..."的状态
        current_text = self.chat_box.get("0.0", "end")
        new_text = current_text.replace("🤖 农智AI助教：[正在思考中...]\n", f"🤖 农智AI助教：\n{reply}\n\n")
        
        self.chat_box.delete("0.0", "end")
        self.chat_box.insert("end", new_text)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def show_sandbox_lab(self):
        self.clear_main_frame()
        title = ctk.CTkLabel(self.main_frame, text="交互编程实验室：农智Jupyter云端沙箱", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        desc = ctk.CTkLabel(self.main_frame, text="【学习与实践环境】免去繁琐的环境配置，提供开启即用、预置计算资源与软件依赖（如PyTorch, Sklearn）的交互式编程体验。\n允许学生一键加载经典农业AI算法模板，快速编写、运行并实时调试代码，实现从理论笔记到真实可复现项目的无缝衔接。", justify="left", font=ctk.CTkFont(size=14))
        desc.grid(row=1, column=0, padx=20, pady=5, sticky="nw")

        # Top Control
        controls_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        controls_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")

        ctk.CTkLabel(controls_frame, text="选择预置代码模板:").pack(side="left", padx=5)
        
        self.template_var = ctk.StringVar(value="[请选择] 空白沙箱")
        self.template_combo = ctk.CTkComboBox(controls_frame, values=[
            "[请选择] 空白沙箱", 
            "模板1：农业产量线性回归预测 (Scikit-learn)", 
            "模板2：小麦病害CNN网络搭建 (PyTorch)",
            "模板3：农业物联网数据清洗 (Pandas)"
        ], variable=self.template_var, width=320, command=self.load_code_template)
        self.template_combo.pack(side="left", padx=10)

        run_btn = ctk.CTkButton(controls_frame, text="▶ 运行代码 (Run Cell)", fg_color="#228B22", hover_color="#006400", command=self.run_sandbox_code)
        run_btn.pack(side="left", padx=20)

        # Editor Frame
        editor_frame = ctk.CTkFrame(self.main_frame)
        editor_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(3, weight=3)
        
        ctk.CTkLabel(editor_frame, text="输入代码 (In [ ]):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.code_editor = ctk.CTkTextbox(editor_frame, font=ctk.CTkFont(family="Consolas", size=14), wrap="none")
        self.code_editor.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Output Frame
        output_frame = ctk.CTkFrame(self.main_frame)
        output_frame.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(4, weight=2)
        
        ctk.CTkLabel(output_frame, text="输出结果 (Out [ ]):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.output_console = ctk.CTkTextbox(output_frame, font=ctk.CTkFont(family="Consolas", size=14), fg_color="#1E1E1E", text_color="#00FF00", wrap="none")
        self.output_console.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_console.configure(state="disabled")
        
        self.load_code_template()

    def load_code_template(self, *args):
        self.code_editor.delete("0.0", "end")
        choice = self.template_var.get()
        
        if "模板1" in choice:
            code = (
                "# 【预置云端环境】Scikit-learn 已就绪\n"
                "import numpy as np\n"
                "from sklearn.linear_model import LinearRegression\n\n"
                "# 1. 模拟农业特征数据 (降水量mm, 温度℃) 和 产量(kg)\n"
                "X = np.array([[800, 25], [750, 26], [900, 24], [600, 28]])\n"
                "y = np.array([6000, 5800, 6200, 5000])\n\n"
                "# 2. 构建并训练回归模型\n"
                "model = LinearRegression()\n"
                "model.fit(X, y)\n\n"
                "# 3. 预测新环境下的产量\n"
                "new_data = np.array([[850, 25.5]])\n"
                "pred = model.predict(new_data)\n"
                "print(f\"✓ 模型训练完成！决定系数 R²: {model.score(X, y):.4f}\")\n"
                "print(f\"✓ 针对新环境条件 {new_data.tolist()[0]} 的预测产量为: {pred[0]:.2f} kg/ha\")\n"
            )
        elif "模板2" in choice:
            code = (
                "# 【预置云端环境】PyTorch 深度学习框架已就绪\n"
                "import torch\n"
                "import torch.nn as nn\n\n"
                "# 定义一个简单的卷积神经网络 (CNN)，用于提取小麦病斑特征\n"
                "class SimpleCNN(nn.Module):\n"
                "    def __init__(self):\n"
                "        super(SimpleCNN, self).__init__()\n"
                "        # 输入通道3(RGB彩色), 输出通道16(特征图数量), 卷积核3x3\n"
                "        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)\n"
                "        self.relu = nn.ReLU()\n\n"
                "    def forward(self, x):\n"
                "        out = self.conv1(x)\n"
                "        out = self.relu(out)\n"
                "        return out\n\n"
                "# 实例化模型并打印网络结构\n"
                "model = SimpleCNN()\n"
                "print(\"✅ 成功搭建 CNN 模型！网络结构如下：\")\n"
                "print(model)\n\n"
                "# 模拟输入一张 64x64 像素的彩色小麦图像张量 (BatchSize=1, Channels=3)\n"
                "dummy_image = torch.randn(1, 3, 64, 64)\n"
                "output_feature_map = model(dummy_image)\n"
                "print(f\"\\n👉 原始图像张量维度: {list(dummy_image.shape)}\")\n"
                "print(f\"👉 经过卷积层后的特征图维度: {list(output_feature_map.shape)}\")\n"
            )
        elif "模板3" in choice:
            code = (
                "# 【预置云端环境】Pandas 数据科学库已就绪\n"
                "import pandas as pd\n"
                "import numpy as np\n\n"
                "# 模拟一份从大棚物联网传感器采集到的带有缺失值的原始数据\n"
                "data = {\n"
                "    '日期': ['2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04'],\n"
                "    '土壤湿度(%)': [45.2, 42.1, 38.5, np.nan], # 这里存在传感器故障导致的缺失值\n"
                "    '光照强度(Lux)': [2100, 2300, 1800, 2200]\n"
                "}\n"
                "df = pd.DataFrame(data)\n\n"
                "print(\"--- 1. 原始物联网传感器数据 ---\")\n"
                "print(df)\n\n"
                "# 数据清洗核心步骤：使用前三天的平均值填充缺失值\n"
                "df['土壤湿度(%)'] = df['土壤湿度(%)'].fillna(df['土壤湿度(%)'].mean())\n"
                "print(\"\\n--- 2. 清洗填补缺失值后的干净数据 ---\")\n"
                "print(df)\n"
            )
        else:
            code = (
                "# 欢迎来到交互式编程云端沙箱！\n"
                "# 在这里，Python、Numpy、Pandas、Scikit-learn、PyTorch 等环境均已预置。\n"
                "# 您可以自由编写代码，并通过上方的【▶ 运行代码】按钮实时查看输出结果。\n\n"
                "print(\"Hello, Agri-AI Explorer!\")\n"
            )
            
        self.code_editor.insert("0.0", code)

    def run_sandbox_code(self):
        import io
        import traceback
        import contextlib

        code = self.code_editor.get("0.0", "end-1c")
        self.output_console.configure(state="normal")
        self.output_console.delete("0.0", "end")
        self.output_console.insert("end", "执行中...\n")
        self.output_console.update()
        self.output_console.delete("0.0", "end")
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                # 使用局部的字典作为运行环境，防止污染主程序的全局变量
                exec_globals = {"__name__": "__main__"}
                exec(code, exec_globals)
                
            output = stdout_buffer.getvalue()
            errors = stderr_buffer.getvalue()
            
            if output:
                self.output_console.insert("end", output)
            if errors:
                self.output_console.insert("end", errors)
                
        except Exception as e:
            err_msg = traceback.format_exc()
            self.output_console.insert("end", err_msg)
            
        self.output_console.insert("end", "\n[程序执行完毕]")
        self.output_console.see("end")
        self.output_console.configure(state="disabled")

    def show_simulation_lab(self):
        self.clear_main_frame()
        title = ctk.CTkLabel(self.main_frame, text="仿真实验平台：农机自动驾驶 (路径规划与避障)", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        desc = ctk.CTkLabel(self.main_frame, text="【仿真与物理引擎】借助虚拟二维网格和传感器模拟，为农业机器人（如无人拖拉机）提供安全、可控、可重复的测试环境。\n学生可以实时观察启发式搜索算法（如 A*）如何在充满随机障碍物（水坑、巨石）的复杂农田中进行全局路径规划和局部避障，\n有效降低真实硬件调试的成本与风险，支持从算法仿真到实际部署的研究流程。", justify="left", font=ctk.CTkFont(size=14))
        desc.grid(row=1, column=0, padx=20, pady=5, sticky="nw")

        # Layout for Simulation
        content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        # Left panel: Simulation Canvas
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        
        self.fig_sim, self.ax_sim = plt.subplots(figsize=(6, 6))
        self.fig_sim.patch.set_facecolor('white')
        self.ax_sim.set_facecolor('#E8F5E9')  # Light green background for farm
        self.ax_sim.set_xticks([])
        self.ax_sim.set_yticks([])
        
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=left_panel)
        self.canvas_sim.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Right panel: Controls
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.grid(row=0, column=1, padx=(10, 0), sticky="nsew")

        ctk.CTkLabel(right_panel, text="虚拟环境控制台", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 10))
        
        ctk.CTkLabel(right_panel, text="障碍物密度 (环境复杂度):").pack(anchor="w", padx=20, pady=(10, 0))
        self.obstacle_slider = ctk.CTkSlider(right_panel, from_=5, to=40, command=self.reset_simulation)
        self.obstacle_slider.set(20)
        self.obstacle_slider.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(right_panel, text="选择控制算法:").pack(anchor="w", padx=20, pady=(10, 0))
        self.sim_algo_var = ctk.StringVar(value="A* 全局路径规划")
        algo_combo = ctk.CTkComboBox(right_panel, values=["A* 全局路径规划", "贪心启发式搜索"], variable=self.sim_algo_var)
        algo_combo.pack(fill="x", padx=20, pady=10)

        reset_btn = ctk.CTkButton(right_panel, text="🔄 重置仿真环境", fg_color="#4682B4", hover_color="#4169E1", command=self.reset_simulation)
        reset_btn.pack(fill="x", padx=20, pady=20)

        self.start_sim_btn = ctk.CTkButton(right_panel, text="▶ 启动自动驾驶仿真", fg_color="#D2691E", hover_color="#A0522D", command=self.start_simulation)
        self.start_sim_btn.pack(fill="x", padx=20, pady=10)
        
        self.sim_status = ctk.CTkLabel(right_panel, text="状态: 等待初始化", font=ctk.CTkFont(size=14), text_color="gray")
        self.sim_status.pack(pady=20)

        self.grid_size = 20
        self.reset_simulation()

    def reset_simulation(self, *args):
        self.start_sim_btn.configure(state="normal")
        self.sim_status.configure(text="状态: 环境已重置，等待运行", text_color="gray")
        
        density = int(self.obstacle_slider.get())
        self.grid_map = np.zeros((self.grid_size, self.grid_size))
        
        # Generate random obstacles
        np.random.seed(int(time.time()) % 10000)
        num_obstacles = int((self.grid_size ** 2) * (density / 100.0))
        
        self.start_pos = (1, 1)
        self.goal_pos = (self.grid_size - 2, self.grid_size - 2)
        
        count = 0
        while count < num_obstacles:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                self.grid_map[r, c] = 1 # 1 represents obstacle
                count += 1
                
        self.draw_simulation_state(agent_pos=self.start_pos, path=[])

    def draw_simulation_state(self, agent_pos, path):
        self.ax_sim.clear()
        self.ax_sim.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax_sim.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax_sim.invert_yaxis()
        
        # Draw grid
        for i in range(self.grid_size + 1):
            self.ax_sim.axhline(i - 0.5, color='white', linewidth=1)
            self.ax_sim.axvline(i - 0.5, color='white', linewidth=1)
            
        # Draw obstacles (Rocks/Puddles)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid_map[r, c] == 1:
                    rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#696969')
                    self.ax_sim.add_patch(rect)
                    
        # Draw goal (Harvest Area)
        goal_rect = plt.Rectangle((self.goal_pos[1] - 0.5, self.goal_pos[0] - 0.5), 1, 1, facecolor='#FFD700', edgecolor='orange', linewidth=2)
        self.ax_sim.add_patch(goal_rect)
        self.ax_sim.text(self.goal_pos[1], self.goal_pos[0], '★', ha='center', va='center', color='red', fontsize=20)
        
        # Draw path
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            self.ax_sim.plot(path_x, path_y, color='blue', linewidth=2, linestyle='--', alpha=0.6)
            
        # Draw agent (Tractor)
        agent_circle = plt.Circle((agent_pos[1], agent_pos[0]), 0.4, color='#FF4500')
        self.ax_sim.add_patch(agent_circle)
        
        # Simulate sensor range (Lidar/Vision Cone)
        sensor_circle = plt.Circle((agent_pos[1], agent_pos[0]), 2.5, color='cyan', alpha=0.2)
        self.ax_sim.add_patch(sensor_circle)

        self.ax_sim.set_title("农田环境：绿色=可行区域，灰色=障碍物，橙色=农机，★=目标点", fontsize=11)
        self.canvas_sim.draw()

    def start_simulation(self):
        self.start_sim_btn.configure(state="disabled")
        self.sim_status.configure(text="状态: 正在计算最优路径...", text_color="blue")
        
        algo = self.sim_algo_var.get()
        use_astar = "A*" in algo
        
        # A* or Greedy Pathfinding in background
        def run_sim():
            path = self.calculate_path(use_astar)
            if not path:
                self.sim_status.configure(text="状态: 无法到达目标！障碍物太多。", text_color="red")
                self.start_sim_btn.configure(state="normal")
                return
                
            self.sim_status.configure(text="状态: 路径已找到，仿真运行中...", text_color="#D2691E")
            
            # Animate movement
            current_path = []
            for step in path:
                current_path.append(step)
                self.draw_simulation_state(agent_pos=step, path=current_path)
                time.sleep(0.15) # Simulation speed
                
            self.sim_status.configure(text="状态: 任务完成！成功到达目标点。", text_color="green")
            self.start_sim_btn.configure(state="normal")

        threading.Thread(target=run_sim, daemon=True).start()

    def calculate_path(self, use_astar):
        import heapq
        
        def heuristic(a, b):
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        neighbors = [(0,1),(0,-1),(1,0),(-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        frontier = []
        heapq.heappush(frontier, (0, self.start_pos))
        came_from = {self.start_pos: None}
        cost_so_far = {self.start_pos: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == self.goal_pos:
                break
                
            for dx, dy in neighbors:
                next_node = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if 0 <= next_node[0] < self.grid_size and 0 <= next_node[1] < self.grid_size:
                    # Check obstacle
                    if self.grid_map[next_node[0], next_node[1]] == 1:
                        continue
                        
                    new_cost = cost_so_far[current] + (1.414 if dx != 0 and dy != 0 else 1)
                    
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + heuristic(self.goal_pos, next_node) if use_astar else heuristic(self.goal_pos, next_node)
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current
                        
        if self.goal_pos not in came_from:
            return None # No path found
            
        # Reconstruct path
        current = self.goal_pos
        path = []
        while current != self.start_pos:
            path.append(current)
            current = came_from[current]
        path.append(self.start_pos)
        path.reverse()
        return path

if __name__ == "__main__":
    app = App()
    app.mainloop()
