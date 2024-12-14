from huggingface_hub import snapshot_download


model_name1 = "kalyan99/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
model_name2 = "kalyan99/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
model_name3 = "kalyan99/Llama-3.1-8B-Instruct-Q4_K_M.gguf"
path_model = "./models"

snapshot_download(repo_id=model_name1, local_dir=path_model)
snapshot_download(repo_id=model_name2, local_dir=path_model)
snapshot_download(repo_id=model_name3, local_dir=path_model)


