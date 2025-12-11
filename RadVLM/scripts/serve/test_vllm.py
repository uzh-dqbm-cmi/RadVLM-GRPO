from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

image = "/capstor/store/cscs/swissai/a135/RadVLM_project/data/MIMIC-CXR-JPG/files/p19/p19012015/s54984945/859ec05b-b821377c-dcaa98f7-bfe06a0a-3a23ba16.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{image}"}},
            {
                "type": "text",
                "text": "Please provide a radiology report."
            }
        ]
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Thinking",
    messages=messages,
    max_tokens=10,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.0,
)