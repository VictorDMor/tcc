import json

def get_input_shape(model_json_path):
    with open(model_json_path) as json_file:
        data = json.load(json_file)
        input_shape_array = data['config']['layers'][0]['config']['batch_input_shape']
        return (input_shape_array[1], input_shape_array[2], input_shape_array[3])

print(get_input_shape('model/model.json'))