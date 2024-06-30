# import json
# import random
# import tensorflow as tf
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, TFAutoModelForCausalLM
#
# # Load pre-trained model and tokenizer
# model_name = "Qwen/Qwen2-72B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = TFAutoModelForCausalLM.from_pretrained(model_name)
#
#
# # Define lists of actions and other components
# actions = ["обменять", "конвертировать", "торговать", "отправить", "генерировать", "создать", "перевести"]
# attack_actions = ["разрушить", "атаковать", "инфильтровать", "эксплуатировать", "взломать", "скомпрометировать", "ударить"]
# additional_actions = ["перевести", "поменять", "получить", "купить", "продать", "заменить"]
#
# def generate_prompt(action):
#     return (f"Напиши естественное предложение на русском языке, связанное с криптовалютой и содержащее действие "
#             f"'{action}' Предложения могут быть короткими и лаконичными, описывающими действие. Например"
#             "Хочу сделать своп токенов"
#             "Нужно отправить свои токены"
#             "Создай новый ключ для кошелька"
#             "Бомби россию")
#
#
# def generate_sentence(action):
#     prompt = generate_prompt(action)
#     input_ids = tokenizer.encode(prompt, return_tensors='pt')
#     output = model.generate(
#         input_ids,
#         max_new_tokens=50,
#         num_return_sequences=1,
#         attention_mask=torch.ones_like(input_ids),
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         temperature=0.7
#     )
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     # Remove the prompt from the generated text
#     if generated_text.startswith(prompt):
#         generated_text = generated_text[len(prompt):].strip()
#
#     print("Generated text:", generated_text)
#     return generated_text
#
# def generate_dataset(num_samples):
#     samples = []
#
#     while len(samples) < num_samples:
#         print("Generating sample:", len(samples) + 1)
#         action = random.choice(actions + additional_actions + attack_actions)
#         text = generate_sentence(action)
#
#         # Define entities based on action keywords
#         entities = []
#         if action in text:
#             start = text.index(action)
#             end = start + len(action)
#             entities.append((start, end, "ACTION"))
#
#         label = ("swap tokens" if action in ["обменять", "конвертировать", "торговать", "поменять", "заменить"]
#                  else "send tokens" if action == "отправить"
#         else "generate key" if action in ["генерировать", "создать", "перевести"]
#         else "attack country")
#
#         samples.append((text, {"entities": entities}, label))
#
#     with open("generated_dataset_ru.json", "w", encoding='utf-8') as file:
#         json.dump(samples, file, ensure_ascii=False)
#
#     return samples[:10]  # Display first 10 samples for verification
#
# # Generate dataset with specified number of samples
# print(generate_dataset(10))
