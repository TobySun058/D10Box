import gradio as gr
import requests

def query_internvl(image):
    response = requests.post(
        "http://localhost:5000/predict",
        files={"image": image}
    )
    return response.json().get("response", "No response")

demo = gr.Interface(
    fn=query_internvl,
    inputs=gr.Image(type="filepath", label="Upload ADA Drawing (.png)"),
    outputs="text",
    title="ADA Sign Detection (InternVL)",
    description="Upload an architectural drawing and detect ADA-compliant signs with page number and position."
)

if __name__ == "__main__":
    demo.launch()
