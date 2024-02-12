import cv2

def dividir_video(video_path, inicio, fin, nuevo_nombre):
    # Abre el video
    video = cv2.VideoCapture(video_path)

    # Obtiene la información del video (ancho, alto, FPS)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calcula el número de frames para el inicio y fin
    frame_inicio = int(fps * inicio)
    frame_fin = int(fps * fin)

    # Inicializa el escritor de video para el nuevo segmento
    writer = cv2.VideoWriter(nuevo_nombre, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Lee los frames del video original y escribe los frames deseados en el nuevo video
    for i in range(frame_inicio, frame_fin):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            writer.write(frame)

    # Libera los recursos
    video.release()
    writer.release()