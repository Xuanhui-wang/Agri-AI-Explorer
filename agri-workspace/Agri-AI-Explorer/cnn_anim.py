from manim import *
import numpy as np
from PIL import Image
import os

class CNNWheatDisease(Scene):
    def construct(self):
        # 设置白色背景
        self.camera.background_color = WHITE
        
        # 1. 标题 (改为黑色加粗)
        title = Text("CNN如何提取小麦赤霉病(FHB)特征", font="Microsoft YaHei", font_size=36, color=BLACK, weight=BOLD)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP).scale(0.8))

        # 2. 引入真实小麦图片
        # 使用用户提供的中度发病图片
        img_path = r"D:\3.创AI案例征集指南与模板\图片(1)\中度\豫农912.jpg"
        
        try:
            # Manim ImageMobject
            wheat_img = ImageMobject(img_path)
            wheat_img.scale(3.0)
            wheat_img.to_edge(LEFT, buff=1.5)
            
            img_title = Text("真实的患病小麦图像 (豫农912中度)", font="Microsoft YaHei", font_size=24, color=BLACK, weight=BOLD).next_to(wheat_img, UP)
            self.play(FadeIn(wheat_img), Write(img_title))
            self.wait(1)
            
            # 3. 放大局部特征
            # 用一个红色的方框圈出病斑边缘
            box = Square(side_length=0.8, color=RED, stroke_width=4)
            box.move_to(wheat_img.get_center() + RIGHT * 0.8 + UP * 0.2) # 框住一个病斑区域
            
            box_label = Text("提取这5x5像素区域的灰度矩阵", font="Microsoft YaHei", font_size=20, color=RED, weight=BOLD).next_to(box, RIGHT)
            self.play(Create(box), Write(box_label))
            self.wait(2)
            
            self.play(FadeOut(wheat_img), FadeOut(img_title), FadeOut(box), FadeOut(box_label))
            
        except Exception as e:
            print("Error loading image:", e)
            pass

        # 4. 展示这 5x5 像素对应的真实灰度值矩阵
        # 模拟提取到的真实灰度值矩阵 (0-255)
        input_matrix = [
            [85,  88,  90,  92,  85],
            [88, 120, 180, 190,  90],
            [90, 190, 240, 230, 120],
            [85, 180, 230, 180,  88],
            [80,  85,  90,  88,  82]
        ]
        
        # 边缘检测核
        kernel_matrix = [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]
        
        output_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]

        def create_matrix_visual(matrix, title_text, is_kernel=False):
            group = VGroup()
            title = Text(title_text, font="Microsoft YaHei", font_size=20, color=BLACK, weight=BOLD)
            group.add(title)
            
            rows = len(matrix)
            cols = len(matrix[0])
            
            cells = VGroup()
            for r in range(rows):
                for c in range(cols):
                    # 修改文字颜色以在白色背景上突出
                    val = matrix[r][c]
                    color = BLACK
                    if not is_kernel:
                        if val > 200: color = "#D32F2F" # 深红色
                        elif val > 150: color = "#E65100" # 深橙色
                        else: color = "#2E7D32" # 深绿色
                    else:
                        if val > 0: color = "#C62828" # 红色
                        elif val < 0: color = "#1565C0" # 蓝色
                        
                    cell_text = Text(str(val), font="Arial", font_size=20, color=color, weight=BOLD)
                    cell_text.move_to(RIGHT * c * 0.8 + DOWN * r * 0.8)
                    cells.add(cell_text)
            
            # 矩阵括号颜色修改为黑色
            left_bracket = Line(UP * 0.4, DOWN * (rows-0.5)*0.8, color=BLACK).shift(LEFT * 0.5)
            left_bracket.add(Line(left_bracket.get_start(), left_bracket.get_start() + RIGHT * 0.2, color=BLACK))
            left_bracket.add(Line(left_bracket.get_end(), left_bracket.get_end() + RIGHT * 0.2, color=BLACK))
            
            right_bracket = Line(UP * 0.4, DOWN * (rows-0.5)*0.8, color=BLACK).shift(RIGHT * ((cols-1)*0.8 + 0.5))
            right_bracket.add(Line(right_bracket.get_start(), right_bracket.get_start() + LEFT * 0.2, color=BLACK))
            right_bracket.add(Line(right_bracket.get_end(), right_bracket.get_end() + LEFT * 0.2, color=BLACK))
            
            cells.add(left_bracket, right_bracket)
            cells.center()
            cells.next_to(title, DOWN, buff=0.5)
            
            group.add(cells)
            group.cells = cells
            group.entries = [cells[i] for i in range(rows*cols)]
            return group

        # 调整标题文本排版避免重叠
        input_mobject = create_matrix_visual(input_matrix, "小麦局部像素灰度值\n(深红/橙:病斑, 绿:健康)")
        kernel_mobject = create_matrix_visual(kernel_matrix, "边缘检测卷积核\n(中心强化，四周削弱)", True)
        output_mobject = create_matrix_visual(output_matrix, "提取出的特征图\n(突出边缘特征)")

        input_mobject.shift(LEFT * 4 + DOWN * 0.5)
        kernel_mobject.shift(DOWN * 0.5)
        output_mobject.shift(RIGHT * 4 + DOWN * 0.5)

        times_sign = Text("×", font_size=40, color=BLACK, weight=BOLD).move_to(LEFT * 2 + DOWN * 0.5)
        equals_sign = Text("=", font_size=40, color=BLACK, weight=BOLD).move_to(RIGHT * 2 + DOWN * 0.5)

        self.play(
            Create(input_mobject),
            Create(kernel_mobject),
            Create(output_mobject),
            Write(times_sign), Write(equals_sign)
        )
        self.wait(1)

        # 5. 动画演示滑动卷积
        # 鲜艳的红色矩阵框
        window_rect = SurroundingRectangle(
            VGroup(
                input_mobject.entries[0], 
                input_mobject.entries[2], 
                input_mobject.entries[10], 
                input_mobject.entries[12]
            ), 
            color=RED, stroke_width=4, buff=0.2
        )
        
        output_rect = SurroundingRectangle(output_mobject.entries[0], color="#2E7D32", stroke_width=4, buff=0.2)
        calculation_text = Text("", font="Microsoft YaHei", font_size=20, color=BLACK, weight=BOLD).to_edge(DOWN).shift(UP * 0.5)
        self.add(calculation_text)

        # 计算真实结果
        real_output = np.zeros((3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                val = 0
                for kr in range(3):
                    for kc in range(3):
                        val += input_matrix[r+kr][c+kc] * kernel_matrix[kr][kc]
                real_output[r][c] = val

        self.play(Create(window_rect), Create(output_rect))

        # 演示关键步骤
        steps = [(0, 0), (1, 1), (2, 2)]

        for out_row, out_col in steps:
            start_idx = out_row * 5 + out_col
            group_to_surround = VGroup(
                input_mobject.entries[start_idx],
                input_mobject.entries[start_idx + 2],
                input_mobject.entries[start_idx + 10],
                input_mobject.entries[start_idx + 12]
            )
            
            self.play(
                window_rect.animate.move_to(group_to_surround.get_center()),
                output_rect.animate.move_to(output_mobject.entries[out_row * 3 + out_col].get_center()),
                run_time=0.8
            )

            res = real_output[out_row][out_col]
            
            calc_str = "计算矩阵乘法求和: "
            if out_row == 0 and out_col == 0:
                calc_str += "滑过【健康区域】：低灰度值相减后结果较小"
            elif out_row == 1 and out_col == 1:
                calc_str += "滑过【病斑中心】：高灰度减低灰度，得出极大值！"
            elif out_row == 2 and out_col == 2:
                calc_str += "滑过【病斑边缘】：计算局部对比度..."

            # 避免重叠
            new_calc_text = Text(calc_str, font="Microsoft YaHei", font_size=22, color=RED, weight=BOLD).to_edge(DOWN).shift(UP * 0.5)
            self.play(Transform(calculation_text, new_calc_text), run_time=0.5)

            # 更新输出
            color = "#D32F2F" if res > 200 else ("#E65100" if res > 50 else BLACK)
            new_entry = Text(str(res), font="Arial", font_size=24, color=color, weight=BOLD).move_to(output_mobject.entries[out_row * 3 + out_col].get_center())
            self.play(Transform(output_mobject.entries[out_row * 3 + out_col], new_entry), run_time=0.3)
            
            if out_row == 1 and out_col == 1:
                self.wait(1.5)
            else:
                self.wait(0.5)

        # 补全剩余
        self.play(FadeOut(window_rect), FadeOut(output_rect), FadeOut(calculation_text))
        for r in range(3):
            for c in range(3):
                if (r, c) not in steps:
                    res = real_output[r][c]
                    color = "#D32F2F" if res > 200 else ("#E65100" if res > 50 else BLACK)
                    new_entry = Text(str(res), font="Arial", font_size=24, color=color, weight=BOLD).move_to(output_mobject.entries[r * 3 + c].get_center())
                    self.play(Transform(output_mobject.entries[r * 3 + c], new_entry), run_time=0.05)

        summary = Text("结论：卷积核在赤霉病斑(高亮)和健康叶片(暗)交界处计算出极大值，\n从而让AI模型成功'看到'并提取了病斑的边缘特征！", font="Microsoft YaHei", font_size=22, color="#2E7D32", weight=BOLD)
        summary.to_edge(DOWN).shift(UP * 0.5)
        
        self.play(Write(summary))
        self.wait(4)
