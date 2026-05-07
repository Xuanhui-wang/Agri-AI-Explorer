import os
import time
import subprocess
import pyautogui
from pathlib import Path

# Ensure output directory exists
project_root = Path(__file__).resolve().parent
output_dir = project_root / "_screenshots"
output_dir.mkdir(parents=True, exist_ok=True)

# Path to the main application
app_path = project_root / "main.py"

# Start the application
print("Starting Agri-AI Explorer...")
process = subprocess.Popen(["python", app_path])

# Wait for the app to open and render
time.sleep(5)
print("Application started. Waiting for window to settle...")

# We are going to maximize the window to ensure consistent screenshots
# Assuming the window has focus
pyautogui.press('win')
pyautogui.press('up') # Maximize on Windows
time.sleep(2)

def take_screenshot(name):
    path = output_dir / f"{name}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    print(f"Saved screenshot: {path}")

def click_sidebar_button(index):
    # This is a rough estimation of coordinates. 
    # Since we can't reliably know the exact pixel coordinates without UI inspection,
    # we will try to click based on relative positions assuming a maximized 1080p screen.
    # The sidebar is on the left.
    # Button 1 is usually around y=100-150, subsequent buttons are spaced vertically.
    x_pos = 120 # x coordinate inside the sidebar
    y_start = 180
    y_gap = 60
    
    y_pos = y_start + (index - 1) * y_gap
    pyautogui.click(x=x_pos, y=y_pos)
    time.sleep(3) # Wait for animation or rendering to complete

# 1. Feature Extraction (CNN) - Default view when opened
take_screenshot("1_特征透视_病斑提取")

# 2. Regression Analysis
click_sidebar_button(2)
take_screenshot("2_回归分析_产量预测")

# 3. Model Training
click_sidebar_button(3)
# Click "Start Training" to get the graph rendering
# Button is roughly in the top right quadrant of the controls frame
pyautogui.click(x=800, y=200) 
time.sleep(4) # Wait for some training iterations to draw the curve
take_screenshot("3_训练可视_模型调参")

# 4. AI Assistant
click_sidebar_button(4)
# Type something in the chat box
pyautogui.click(x=400, y=800) # Click entry box
pyautogui.write("请解释一下卷积网络中的学习率代表什么？", interval=0.05)
pyautogui.press('enter')
time.sleep(5) # Wait for AI response
take_screenshot("4_智能答疑_助教中心")

# 5. Sandbox
click_sidebar_button(5)
# Click Run Code
pyautogui.click(x=500, y=180) # Click run button
time.sleep(2)
take_screenshot("5_交互编程_农智沙箱")

# 6. Simulation
click_sidebar_button(6)
# Click Start Simulation
pyautogui.click(x=900, y=550) # Click start button
time.sleep(4) # Wait for some pathfinding to draw
take_screenshot("6_仿真实验_农机驾驶")

# Close the application
process.terminate()
print("Screenshots completed successfully.")
