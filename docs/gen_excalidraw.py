import json
import random
import uuid
import time

class ExcalidrawGenerator:
    def __init__(self):
        self.elements = []
        self.font_family = 1  # 1: Virgil (Hand-drawn), 2: Normal, 3: Cascadia
        
    def _get_common_props(self, x, y, width, height, stroke_color="#000000", bg_color="transparent"):
        return {
            "id": str(uuid.uuid4()),
            "type": "rectangle",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "angle": 0,
            "strokeColor": stroke_color,
            "backgroundColor": bg_color,
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "roundness": { "type": 3 },
            "seed": random.randint(1000, 99999),
            "version": 1,
            "versionNonce": random.randint(0, 1000000),
            "isDeleted": False,
            "boundElements": [],
            "updated": int(time.time() * 1000),
            "link": None,
            "locked": False,
        }

    def add_rect(self, x, y, width, height, label=None, bg_color="#transparent", stroke_color="#000000", font_size=16):
        rect_id = str(uuid.uuid4())
        rect = self._get_common_props(x, y, width, height, stroke_color, bg_color)
        rect["id"] = rect_id
        rect["type"] = "rectangle"
        self.elements.append(rect)
        
        if label:
            lines = label.split('\n')
            num_lines = len(lines)
            
            # Use smaller font size for multi-line labels to fit better
            final_font_size = font_size
            if num_lines > 1:
                final_font_size = int(font_size * 0.9) # Slightly smaller text for better fit
            
            # Calculate total text block height
            # Line height factor 1.4 for good spacing
            line_height = final_font_size * 1.4
            total_text_height = num_lines * line_height
            
            # Start Y position to vertically center the block
            # y + (height - total_text_height) / 2 centers the block
            start_y = y + (height - total_text_height) / 2
            
            for i, line in enumerate(lines):
                # Calculate Y for this line
                current_y = start_y + i * line_height
                
                # Use center of the rectangle as the anchor X
                center_x = x + width / 2
                
                self.add_text(center_x, current_y, line, fontSize=final_font_size, start_align="center")
        
        return rect_id

    def add_text(self, x, y, text, fontSize=20, color="#000000", start_align="left"):
        # Rough estimation of dimensions
        width = len(text) * fontSize * 0.6
        height = fontSize * 1.2
        
        if start_align == "center":
            x = x - width / 2
        
        obj = {
            "type": "text",
            "version": 1,
            "versionNonce": 0,
            "isDeleted": False,
            "id": str(uuid.uuid4()),
            "fillStyle": "hachure",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "angle": 0,
            "x": x,
            "y": y,
            "strokeColor": color,
            "backgroundColor": "transparent",
            "width": width,
            "height": height,
            "seed": random.randint(1000, 9999),
            "groupIds": [],
            "roundness": None,
            "boundElements": [],
            "updated": int(time.time()),
            "link": None,
            "locked": False,
            "fontSize": fontSize,
            "fontFamily": self.font_family,
            "text": text,
            "rawText": text,
            "textAlign": "center" if start_align == "center" else "left",
            "verticalAlign": "top",
            "containerId": None,
            "originalText": text
        }
        self.elements.append(obj)
        return obj["id"]

    def add_arrow(self, start_x, start_y, end_x, end_y, stroke_color="#000000"):
        obj = {
            "type": "arrow",
            "version": 1,
            "versionNonce": 0,
            "isDeleted": False,
            "id": str(uuid.uuid4()),
            "fillStyle": "hachure",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "angle": 0,
            "x": start_x,
            "y": start_y,
            "strokeColor": stroke_color,
            "backgroundColor": "transparent",
            "width": abs(end_x - start_x),
            "height": abs(end_y - start_y),
            "seed": random.randint(1000, 9999),
            "groupIds": [],
            "roundness": { "type": 2 },
            "boundElements": [],
            "updated": int(time.time()),
            "link": None,
            "locked": False,
            "startBinding": None,
            "endBinding": None,
            "lastCommittedPoint": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "points": [
                [0, 0],
                [end_x - start_x, end_y - start_y]
            ]
        }
        self.elements.append(obj)

    def save(self, filename):
        data = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {
                "gridSize": None,
                "viewBackgroundColor": "#ffffff"
            },
            "files": {}
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def generate_nanochat_arch():
    gen = ExcalidrawGenerator()
    
    # Title
    gen.add_text(1000, 50, "NanoChat 代码架构图", fontSize=36, start_align="center")
    
    # --- Layer 1: Scripts Layer ---
    layer1_y = 120
    container_width = 1800 # 增加总宽度
    
    gen.add_text(100, layer1_y - 40, "入口脚本层 (scripts/)", fontSize=24)
    # Container
    gen.add_rect(80, layer1_y, container_width, 180, bg_color="#dae8fc", stroke_color="#6c8ebf")
    
    scripts = [
        ("base_train.py\n预训练", "#fff2cc"),
        ("base_eval.py\n基座评估", "#fff2cc"),
        ("chat_sft.py\nSFT微调", "#d5e8d4"),
        ("chat_rl.py\n强化学习", "#d5e8d4"),
        ("chat_eval.py\n对话评估", "#d5e8d4"),
        ("chat_web.py\nWeb服务", "#e1d5e7"),
        ("chat_cli.py\nCLI交互", "#e1d5e7"),
        ("tok_train.py\n分词训练", "#f8cecc"),
        ("tok_eval.py\n分词评估", "#f8cecc"),
    ]
    
    # 增加间距
    sx = 140
    item_width = 140
    item_height = 90
    spacing = 40
    
    for name, color in scripts:
        gen.add_rect(sx, layer1_y + 45, item_width, item_height, label=name, bg_color=color, font_size=16)
        sx += item_width + spacing

    # --- Layer 2: Core Layer ---
    layer2_y = 380 # 下移
    gen.add_text(100, layer2_y - 40, "核心模块层 (nanochat/)", fontSize=24)
    # Container
    gen.add_rect(80, layer2_y, container_width, 360, bg_color="#ffe6cc", stroke_color="#d79b00")
    
    # Groups
    group_width = 380 # 增加组宽度
    group_spacing = 60
    
    groups = [
        ("模型架构", "#fff2cc", group_width, [
            ("gpt.py\nGPT Model", "#ffffff", 20, 60),
            ("flash_attn.py\nFA3 / SDPA", "#ffffff", 200, 60),
            ("optim.py\nMuonAdamW", "#ffffff", 20, 180),
            ("ckpt_manager.py\nSave / Load", "#ffffff", 200, 180) # 缩短名称防止溢出
        ]),
        ("数据处理", "#d5e8d4", group_width, [
            ("tokenizer.py\nBPE Tokenizer", "#ffffff", 20, 60),
            ("dataloader.py\nDistLoader", "#ffffff", 200, 60),
            ("dataset.py\nDataset Utils", "#ffffff", 20, 180)
        ]),
        ("推理引擎", "#e1d5e7", group_width, [
            ("engine.py\nKVCache/Gen", "#ffffff", 20, 60),
            ("execution.py\nPython Tool", "#ffffff", 200, 60),
            ("ui.html\nFrontend", "#ffffff", 20, 180)
        ]),
        ("评估模块", "#f8cecc", group_width, [
            ("core_eval.py\nCORE Score", "#ffffff", 20, 60),
            ("loss_eval.py\nBPB Eval", "#ffffff", 200, 60),
            ("report.py\nReport Gen", "#ffffff", 20, 180),
            ("common.py\nUtils", "#ffffff", 200, 180)
        ])
    ]
    
    start_x = 120
    inner_item_width = 150
    inner_item_height = 80
    
    for g_name, g_color, g_width, items in groups:
        # Group Box
        gen.add_rect(start_x, layer2_y + 50, g_width, 280, label=None, bg_color=g_color)
        gen.add_text(start_x + 20, layer2_y + 60, g_name, fontSize=20)
        
        for item_name, item_color, ix, iy in items:
            # ix is relative offset, iy is relative offset
            gen.add_rect(start_x + ix, layer2_y + 50 + iy, inner_item_width, inner_item_height, label=item_name, bg_color=item_color, font_size=14)
            
        start_x += g_width + group_spacing
        
    # --- Layer 3: Tasks Layer ---
    layer3_y = 800 # 下移
    gen.add_text(100, layer3_y - 40, "任务数据集层 (tasks/)", fontSize=24)
    # Container
    gen.add_rect(80, layer3_y, container_width, 160, bg_color="#f5f5f5", stroke_color="#666666")
    
    tasks = [
        ("common.py", "#ffffff"),
        ("smoltalk.py", "#ffffff"),
        ("mmlu.py", "#ffffff"),
        ("arc.py", "#ffffff"),
        ("gsm8k.py", "#ffffff"),
        ("humaneval.py", "#ffffff"),
        ("spellingbee.py", "#ffffff"),
        ("customjson.py", "#ffffff"),
    ]
    
    tx = 140
    task_width_item = 160
    task_spacing = 40
    
    for name, color in tasks:
        gen.add_rect(tx, layer3_y + 50, task_width_item, 70, label=name, bg_color=color, font_size=18)
        tx += task_width_item + task_spacing
        
    # Arrows
    gen.add_arrow(200, layer1_y + 180, 200, layer2_y)
    gen.add_arrow(1000, layer1_y + 180, 1000, layer2_y)
    
    gen.add_arrow(500, layer2_y + 360, 500, layer3_y)

    gen.save("c:/projects/nanochat/docs/nanochat_architecture.excalidraw")

if __name__ == "__main__":
    generate_nanochat_arch()
