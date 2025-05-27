from datasets import Dataset

def load_ada_sign_dataset():
    base_path = "./drawings/ADA_Examples"
    samples = [
        {
            "image_path": f"{base_path}/ADA_Signs_1.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "room_number", "text_or_symbols": "ROOM NAME with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_2.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "occupancy_load", "text_or_symbols": "OCCUPANCY LOAD with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_3.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "restroom", "text_or_symbols": "MEN with wheelchair icon and Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_4.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "occupancy_load", "text_or_symbols": "OCCUPANCY LOAD with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_5.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "room_number", "text_or_symbols": "C233 with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_6.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "room_number", "text_or_symbols": "ROOM NAME with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_7.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "restroom", "text_or_symbols": "MEN with wheelchair icon and Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_8.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "room_number", "text_or_symbols": "ROOM # and ROOM NAME with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_9.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "occupancy_load", "text_or_symbols": "NOTICE OCCUPANCY IS LIMITED TO with Braille", "position_in_image": "center"}'
        },
        {
            "image_path": f"{base_path}/ADA_Signs_10.png",
            "instruction": "Identify the ADA-compliant signage design in this architectural example.",
            "response": '{"type": "room_number", "text_or_symbols": "AIRPORT OFFICE 101 with Braille", "position_in_image": "center"}'
        }
    ]
    return Dataset.from_list(samples)
