import onnx
import tensorrt

class ModelFactory:
    @staticmethod
    def create_model(model_type):
        """
        This method creates a model instance based on the given model_type.
        :param model_type: A string representing the type of model to create.
        :return: An instance of the specified model type.
        """
        if model_type == "linear":
            return LinearModel()
        elif model_type == "tree":
            return DecisionTreeModel()
        elif model_type == "neural":
            return NeuralNetworkModel()
        else:
            raise ValueError("Invalid model type specified")

    @staticmethod
    def convert_model(model, output_path):
        """
        This method converts a model to ONNX format, saves it to the specified output path, and converts it to TensorRT format.
        :param model: An instance of a model to convert.
        :param output_path: A string representing the path to save the converted model.
        :return: None
        """
        # Convert model to ONNX format
        onnx_model = onnx.load(model)
        
        # Save ONNX model to output path
        onnx.save(onnx_model, output_path)
        
        # Convert ONNX model to TensorRT format
        trt_model = tensorrt.convert_onnx_model(onnx_model)
        
        # Save TensorRT model to output path
        with open(output_path, "wb") as f:
            f.write(trt_model)

        
          