import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_frame(filename):
    with open(filename, 'r') as f:
        # Lire la première ligne pour obtenir nx et ny
        first_line = f.readline().strip()
        nx, ny = map(int, first_line.split())
        
        # Charger les données en sautant une ligne à chaque "ny" valeurs
        data = np.loadtxt(f).reshape((nx, ny))
    
    return data

def get_frame_filenames(folder):
    # Obtenir la liste des fichiers et les trier simplement par le numéro dans le nom
    files = [file for file in os.listdir(folder) if file.startswith('frame_v_') and file.endswith('.txt')]
    
    # Trier les fichiers par le numéro dans le nom
    files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    return [os.path.join(folder, file) for file in files]


def create_animation(filenames):
    # Charger la première frame pour initialiser la figure
    first_frame = load_frame(filenames[0])
    nx, ny = first_frame.shape
    
    fig, ax = plt.subplots()
    cax = ax.imshow(first_frame, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(cax, label="Concentration")
    
    def update(frame):
        data = load_frame(frame)
        cax.set_array(data)
        ax.set_title(f"Frame: {frame.split('/')[-1]}")
        return cax,

    ani = animation.FuncAnimation(fig, update, frames=filenames, blit=True, interval=1)

    print("Save medias...")
   # Enregistrer l'animation en GIF
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)
    ani.save(media_dir+"GS.gif", writer='pillow', fps=20)

    # Sauvegarder la dernière image en PNG
    last_frame = load_frame(filenames[-1])
    plt.imshow(last_frame, cmap='viridis', vmin=0, vmax=1)
    plt.title(f"Last Frame: {filenames[-1].split('/')[-1]}")
    plt.savefig(media_dir+"GS.png")

    print("Begin Animation...")
    plt.show()

def main():
    folder = 'frames'
    filenames = get_frame_filenames(folder)
    
    if not filenames:
        print(f"No files found in {folder}")
        return
    
    print(f"Found {len(filenames)} files.")
    create_animation(filenames)

if __name__ == '__main__':
    main()

