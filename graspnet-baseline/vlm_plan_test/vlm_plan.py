#!/usr/bin/env python3
import base64
import time  
import cv2
import os
from openai import OpenAI

def convert_image_to_base64(color_image):
    # encode image to JPEG format
    _, buffer = cv2.imencode('.jpg', color_image)
    # convert byte data to Base64 encoding
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

class QwenVLRequester:
    def __init__(self, ignore_former_conversation=False):
        # 初始化 OpenAI 客户端，指向你的中转 API
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1' 
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = 'qwen-vl-plus'  # 使用 Qwen 模型
        
        # 核心技能说明书（保持原样，不随意篡改你的 Prompt 工程）
        self.system_prompt = """
                    Skill Library (The skills you can use to solve the task. ):
                    1 Identifying Object (According to input, this skill can segment the target object in the image. You must generate a JSON object containing the label and the bounding box coordinates of the object. Format: {"name": "label", "bbox": [xmin, ymin, xmax, ymax]}. Coordinates must be integers. Input: None; output: “RGBD image with mask”);
                    2 Generating Grasping Point and Posture (Received the RGBD image with mask from skill 1, this skill can generate the desire pose and position for end effector to grasp the object. Input: “RGBD image with mask”; Output: “desire position and posture”);
                    3 Generating Releasing Point (Received the RGBD image with mask from skill 1, this skill output desire position as  the destination point for end effector moving to.  Input: “RGBD image with mask”; Output: “desire position”);
                    4 Moving Hand (Received the desire pose, this skill can move the hand to the desire position in the specified posture. Input: “desire position and posture”or “desire position”; Output: “None”);
                    5 Grasping Object (Close end effector to grasp the object after skill 4. Prerequisite of this skill is that the hand is not holding other objects. Input: “None”; Output: “None”);
                    6 Releasing Object (Open end effector to release the object after skill 4. Input: “None”; Output: “None“);
                    7 Rotating Hand (The end effector rotate at a fixed angle around a fixed axis, while remaining its' posture. Input: “None”; Output: “None”);
                    
                    Hardware Description(the tools you can use):
                    1 RGBD camera (Installed on the wrist, it can offer RGBD image including RGB and pixel-level depth information);
                    2 Dexterous hand-arm system (Only single dexterous hand-arm system, which means the hand can not grasp the other object during holding one.).
                    
                    Response Format:
                    Give the skill number sequence and add ## among them. For instance: 
                    Final result:
                    ##skill01({"name": "apple", "bbox": [150, 200, 350, 450]})##skill02##skill04##.
                    Note: The input of one skill must match the output of the former one.
                """
        
        self.ignore_former_conversation = ignore_former_conversation

    def request_model(self, user_command, color_image):
        start_time = time.time()
        
        # 动态拼接：用户的真实指令 + 系统的技能约束
        full_question = f"Task Description:\n{user_command}\n\n{self.system_prompt}"
        
        if self.ignore_former_conversation:
            full_question = "Ignore the former conversation.\n" + full_question
            print("已忽略历史对话 (ignore_former_conversation)")

        # 生成请求格式并发送
        messages = self.create_messages(full_question, color_image)
        response_text = ""
        
        try:
            print("模型正在思考中...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                max_tokens=1024
            )
            
            # 流式接收响应
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        response_text += delta_content
                        
        except Exception as e:
            print(f"请求大模型时发生错误: {e}")
            
        print("\n最终生成的任务序列文本:", response_text)
        print(f"执行耗时: {time.time() - start_time:.2f} 秒")
        
        return response_text

    def create_messages(self, question, color_image):
        base64_image = convert_image_to_base64(color_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        return messages