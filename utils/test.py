import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np
def visualize_sample_image(image,true_label,score,threshold,hierarchy_dicts,output_file_path):
    # Festlegen der Größe des Ausgabebildes
    plt.figure(figsize=(8, 6))  # Breite: 8 Zoll, Höhe: 6 Zoll
    thresholded_score = score > threshold
    # Anzeigen des Bildes
    plt.imshow(image)
    plt.axis('off')  # Achsen ausschalten
    swapped_hierarchy_dict = [{v: k for k, v in hierarchy_dict.items()} for hierarchy_dict in hierarchy_dicts]
    # Text für die richtigen Labels
    base_text_anchor = image.size[1] + 20
    start_index = 0
    for i in range(len(swapped_hierarchy_dict)):
        plt.text(0,base_text_anchor,f'Hierarchy-Layer-{i+1}:',fontsize=9,weight='bold')
        anchor_counter = 0
        for j in swapped_hierarchy_dict[i].keys():
            wk_id = swapped_hierarchy_dict[i][j].split('_')[-1]
            if true_label[start_index+j] == 1 and true_label[start_index+j] == thresholded_score[start_index+j]:
                plt.text(35+(anchor_counter+1)*38,base_text_anchor,f'{wk_id}',color='green',fontsize=9)
                anchor_counter+=1
            elif true_label[start_index+j] == 1 and true_label[start_index+j] != thresholded_score[start_index+j]:
                plt.text(35+(anchor_counter+1)*38,base_text_anchor,f'{wk_id}',color='red',fontsize=9)
                anchor_counter+=1
            elif true_label[start_index+j] == 0 and true_label[start_index+j] != thresholded_score[start_index+j]:
                plt.text(35+(anchor_counter+1)*38,base_text_anchor,f'{wk_id}',color='orange',fontsize=9)
                anchor_counter+=1
        base_text_anchor+=10
        start_index+=len(swapped_hierarchy_dict[i])
        
    legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Positive'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Negative'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='False Positive')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_file_path,bbox_inches='tight')
    plt.clf()
image = Image.open('E:/workspace/Hierarchical-Multi-Label-Text-Classification/data/image_harnn/resized_images/224x224/1988_in_Odessa_10.jpg')
with open('E:/workspace/Hierarchical-Multi-Label-Text-Classification/data/image_harnn/einrichtungsgegenstände/filtered_hierarchy_dicts.json', 'r') as infile:
    hierarchy_dicts = json.load(infile)
true_label = np.array([1,0,1,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0])
score = np.array([0.7,0.6,0.3,0.1,0.2,0.45,0.9,0,0.3,0,0,0.6,0.4,0,0,0,0.7,0.9,0.8,0,0,0,0,0])
output_file_path = 'E:/workspace/Hierarchical-Multi-Label-Text-Classification/data/image_harnn/einrichtungsgegenstände/sample01.jpg'
visualize_sample_image(image,true_label,score,threshold=0.5,hierarchy_dicts=hierarchy_dicts,output_file_path=output_file_path)

