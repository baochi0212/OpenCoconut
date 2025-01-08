import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatPipeline:
    def __init__(self, model_name: str):
        """
        Inicializa el modelo y el tokenizer,
        así como la lista de mensajes (messages).
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Mensaje inicial con la instrucción del sistema.
        self.messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Please think step by step."
            }
        ]

    def chat(self, user_prompt: str, max_new_tokens=16384) -> str:
        """
        - Agrega el prompt del usuario a la conversación.
        - Genera la respuesta 'thinking'.
        - Genera la respuesta final como 'assistant'.
        - Retorna la última respuesta (la del 'assistant').
        """

        # 1) Agregamos el prompt del usuario
        self.messages.append({
            "role": "user",
            "content": user_prompt
        })

        # 2) Generamos la respuesta final del 'assistant'
        assistant_response = self._generate_response(
            max_new_tokens=max_new_tokens)
        self.messages.append(
            {"role": "assistant", "content": assistant_response})

        return assistant_response

    def _generate_response(self, max_new_tokens: int = 2500) -> str:
        """
        Método auxiliar que construye el texto a partir de la lista de mensajes,
        lo tokeniza y llama al modelo para generar la siguiente respuesta.
        Devuelve el texto decodificado.
        """
        # Construimos el texto con apply_chat_template (propio de Qwen/SmallThinker)
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Preparamos los tensores para la inferencia
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        # Generamos la respuesta
        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        # Para aislar solo los tokens generados nuevos,
        # cortamos por la longitud de la entrada
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(model_inputs.input_ids, output_ids)
        ]

        # Decodificamos
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response


if __name__ == "__main__":
    # PowerInfer_SmallThinker-3B-Preview_infe_0001
    # Ejemplo de uso:
    model_name = "./checkpoint_spp/SmallThinker-3B-Preview"
    pipeline = ChatPipeline(model_name)

    prompt = "Why VinAI Research good ?"
    final_answer = pipeline.chat(prompt, max_new_tokens=1512)
    print("\n--- Respuesta final ---")
    print(final_answer)
