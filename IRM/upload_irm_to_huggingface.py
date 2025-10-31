from huggingface_hub import upload_folder

repo_id = "ayarnte/Idea_Reward_Model"
local_dir = "./irm_sci_huber_z_splitstats"  


upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model"
)

print("✅ Upload completed:", repo_id)
