import camera
from sys import exit
import cv2
import mediapipe as mp
import numpy as np

if __name__ == "__main__":
    webcam = camera.WebcamStream()
    webcam.start()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.9)

    mp_drawing = mp.solutions.drawing_utils

    while True:
        frame = webcam.read()
        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose tahmini yap
        results = pose.process(greyFrame)

        # Eğer landmark'lar varsa
        if results.pose_landmarks:
            # Anahtar noktaları görüntü üzerine çiz
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Landmark noktalarını al
            landmarks = results.pose_landmarks.landmark

            # Burun (baş) ve omuzlar (gövde) noktaları
            nose = landmarks[0]  # Burun

            left_shoulder = landmarks[11]  # Sol omuz
            right_shoulder = landmarks[12]  # Sağ omuz

            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2

            # Kalçalar
            hip_left = landmarks[24]
            hip_right = landmarks[23]

            hip_mid_x = (hip_left.x + hip_right.x) / 2
            hip_mid_y = (hip_left.y + hip_right.y) / 2

            # Burun ile omuz orta noktası arasında çizgi çiz
            cv2.line(frame, 
                     (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])), 
                     (int(shoulder_mid_x * frame.shape[1]), int(shoulder_mid_y * frame.shape[0])), 
                     (0, 0, 255), 2)

            # Omuz ortasından kalça ortasına doğru çizgi çiz
            cv2.line(frame, 
                     (int(shoulder_mid_x * frame.shape[1]), int(shoulder_mid_y * frame.shape[0])), 
                     (int(hip_mid_x * frame.shape[1]), int(hip_mid_y * frame.shape[0])), 
                     (255, 0, 0), 2)

            # İlk doğru vektörü: (nose -> shoulder_mid)
            vec1 = np.array([
                int(shoulder_mid_x * frame.shape[1]) - int(nose.x * frame.shape[1]), 
                int(shoulder_mid_y * frame.shape[0]) - int(nose.y * frame.shape[0])
            ])

            # İkinci doğru vektörü: (0, 475) -> (635, 475)
            vec2 = np.array([635 - 0, 475 - 475])

            # Üçüncü doğru vektörü (nose -> hip_mid)
            vec3 = np.array([
                int(hip_mid_x * frame.shape[1]) - int(nose.x * frame.shape[1]),
                int(hip_mid_y * frame.shape[0]) - int(nose.y * frame.shape[0])
            ])

            # Nokta çarpımı
            dot_product = np.dot(vec1, vec2)
            dot_product2 = np.dot(vec2, vec3)

            # Vektör büyüklükleri
            mag_vec1 = np.linalg.norm(vec1)
            mag_vec2 = np.linalg.norm(vec2)
            mag_vec3 = np.linalg.norm(vec3)

            # İki doğru arasındaki açıyı hesapla (radyan cinsinden)
            cos_theta = dot_product / (mag_vec1 * mag_vec2)
            theta_radians = np.arccos(cos_theta)

            cos_theta2 = dot_product2 / (mag_vec2 * mag_vec3)
            theta_radians2 = np.arccos(cos_theta2)

            # Açıyı dereceye çevir
            theta_degrees = np.degrees(theta_radians)
            theta_degrees2 = np.degrees(theta_radians2)

            if theta_degrees2 > 160:
                print(f"Düştü : {theta_degrees2}")

        # Yatay referans çizgisi çiz
        cv2.line(frame, (0, 475), (635, 475), (0, 0, 255), 2)  

        # Sonuçları göster
        cv2.imshow('Frame', frame)

        # 'q' tuşuna basıldığında çık
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    webcam.stop(tFlag=True)
    exit(0)




