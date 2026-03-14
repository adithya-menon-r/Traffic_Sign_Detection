import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'Epoch': range(1, 16),
    'Box_Loss': [1.04, 0.9456, 0.9395, 0.8941, 0.8698, 0.8198, 0.7928, 0.7774, 0.7537, 0.7422, 0.7281, 0.7035, 0.6861, 0.6565, 0.6556],
    'Cls_Loss': [2.711, 1.734, 1.608, 1.404, 1.213, 1.281, 1.098, 1.028, 0.9079, 0.8421, 0.7878, 0.7291, 0.6898, 0.6548, 0.6217],
    'DFL_Loss': [1.073, 1.013, 1.036, 1.012, 0.998, 0.9729, 0.9433, 0.9408, 0.9391, 0.9376, 0.9117, 0.9077, 0.9002, 0.902, 0.8921],
    'mAP50':    [0.521, 0.511, 0.675, 0.724, 0.776, 0.791, 0.832, 0.781, 0.84, 0.864, 0.889, 0.841, 0.885, 0.88, 0.891]
}

df = pd.DataFrame(data)

df['Total_Loss'] = df['Box_Loss'] + df['Cls_Loss'] + df['DFL_Loss']

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Total_Loss'], marker='o', linestyle='-', color='#e74c3c', linewidth=2, label='Total Loss')
plt.plot(df['Epoch'], df['Cls_Loss'], marker='x', linestyle='--', color='#e67e22', alpha=0.7, label='Class Loss')
plt.plot(df['Epoch'], df['Box_Loss'], marker='x', linestyle='--', color='#2980b9', alpha=0.7, label='Box Loss')

plt.title('Training Loss over 15 Epochs (YOLOv8)', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss Value (Lower is Better)')
plt.xticks(range(1, 16))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('graph1_training_loss_15epochs.png', dpi=300)
print("Saved 'graph1_training_loss_15epochs.png'")

plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['mAP50'], marker='o', linestyle='-', color='#27ae60', linewidth=3, label='mAP50 (Accuracy)')

best_epoch = df.loc[df['mAP50'].idxmax()]
plt.annotate(f"Peak Accuracy: {best_epoch['mAP50']*100:.1f}%", 
             xy=(best_epoch['Epoch'], best_epoch['mAP50']), 
             xytext=(best_epoch['Epoch']-3, best_epoch['mAP50']-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Detection Accuracy (mAP50) over 15 Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('mAP50 Score (Higher is Better)')
plt.xticks(range(1, 16))
plt.ylim(0.4, 1.0) 
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('graph2_accuracy_map_15epochs.png', dpi=300)
print("Saved 'graph2_accuracy_map_15epochs.png'")