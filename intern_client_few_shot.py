import gradio as gr
import requests
import json

def query_internvl(image_path):
    try:
        # Open the image in binary mode
        with open(image_path, "rb") as f:
            response = requests.post(
                "http://localhost:5000/predict",
                files={"image": ("image.png", f, "image/png")}
            )
        response.raise_for_status()

        # Try to parse JSON
        try:
            data = response.json()
            resp = data.get("response", "No 'response' field in JSON.")
            if isinstance(resp, dict):
                return json.dumps(resp, indent=2)
            return resp
        except ValueError:
            return f"⚠️ Server did not return valid JSON.\n\nRaw response:\n{response.text}"

    except requests.exceptions.RequestException as e:
        return f"🚫 Failed to contact the server.\n\nDetails:\n{e}"

    except Exception as e:
        return f"❌ Unexpected error:\n{e}"

# Launch Gradio interface
demo = gr.Interface(
    fn=query_internvl,
    inputs=gr.Image(type="filepath", label="Upload ADA Drawing (.png)"),
    outputs=gr.Textbox(label="Detection Result"),
    title="ADA Sign Detection (InternVL)",
    description="Upload an architectural drawing (.png) to detect ADA-compliant signs with type, position, and sheet number.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
