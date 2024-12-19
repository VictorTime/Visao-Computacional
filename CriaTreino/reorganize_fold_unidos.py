import os
import shutil
import random

# Caminhos atuais e de destino
data_path = r"D:\4_datasets\yolo_crop_pancreas_no_norm"
new_data_path = "./inputs/mass_separada_3"

# Definir os folds para treino, validação e teste
train_folds = ["1", "2"]
val_fold = ["3"]
test_fold = ["4"]

# Criar a nova estrutura de diretórios
for dataset_type in ["train", "val", "test"]:
    for cls in ["images", "masks"]:
        os.makedirs(os.path.join(new_data_path, dataset_type, cls), exist_ok=True)
        if cls == "masks":
            os.makedirs(os.path.join(new_data_path, dataset_type, cls, "0"), exist_ok=True)

# Função para mover os arquivos para a nova estrutura, verificando correspondência entre slice e label_mass
def move_files(fold_list, dataset_type):
    for fold in fold_list:
        fold_path = os.path.join(data_path, fold)
        if not os.path.isdir(fold_path):
            continue
        
        for patient_id in os.listdir(fold_path):
            patient_path = os.path.join(fold_path, patient_id)
            
            for pancreas_dir in os.listdir(patient_path):
                # pancreas_path = os.path.join(patient_path, pancreas_dir)
                # if not os.path.isdir(pancreas_path):
                #     continue
                
                # Caminhos das pastas label_mass e slice
                label_mass_path = os.path.join(patient_path, "label_mass")
                slice_path = os.path.join(patient_path, "slice")
                
                if os.path.exists(slice_path) and os.path.exists(label_mass_path):
                    slice_files = set(os.listdir(slice_path))
                    label_files = set(os.listdir(label_mass_path))
                    
                    # Encontrar arquivos que estão tanto em slice quanto em label_mass
                    common_files = slice_files.intersection(label_files)
                    
                    # Mover os arquivos comuns para a nova estrutura
                    for filename in common_files:
                        # Mover arquivo de slice para images
                        old_slice_path = os.path.join(slice_path, filename)
                        new_slice_path = os.path.join(new_data_path, dataset_type, "images", filename)
                        shutil.copy(old_slice_path, new_slice_path)
                        
                        # Mover arquivo de label_mass para masks/0
                        old_label_path = os.path.join(label_mass_path, filename)
                        new_label_path = os.path.join(new_data_path, dataset_type, "masks", "0", filename)
                        shutil.copy(old_label_path, new_label_path)

# Mover os arquivos de cada conjunto
move_files(train_folds, "train")
move_files(val_fold, "val")
move_files(test_fold, "test")

print("Reorganização concluída com sucesso!")
