import cv2
import face_recognition
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# کلاس اصلی که استخراج، ارزیابی کیفیت، خوشه‌بندی و نمایش چهره‌ها را انجام می‌دهد
class EnhancedVideoFaceComparator:

    def __init__(self):
        # لیست کدهای شناسه (encodings) چهره‌های شناخته‌شده (بعد از خوشه‌بندی)
        self.known_face_encodings = []
        # لیست شناسه‌های (IDs) تولید شده برای هر فرد
        self.known_face_ids = []
        # شمارنده برای تولید شناسه‌های یکتا
        self.face_counter = 0

    def apply_pca(self, encodings, variance_threshold=0.95):
        """
        اعمال PCA روی encodingهای چهره برای کاهش ابعاد
        """
        if len(encodings) <= 1:
            return encodings
        
        # استانداردسازی داده‌ها
        encodings_standardized = (encodings - np.mean(encodings, axis=0)) / np.std(encodings, axis=0)
        
        # اعمال PCA
        pca = PCA(n_components=variance_threshold, random_state=42)
        encodings_pca = pca.fit_transform(encodings_standardized)
        
        print(f"🔹 PCA: Reduced dimensions from {encodings.shape[1]} to {encodings_pca.shape[1]}")
        print(f"🔹 Explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        return encodings_pca

    def calculate_face_quality(self, face_image, face_location):
        """
        محاسبهٔ کیفیت یک تصویر صورت با چند شاخص ساده:
        - اندازه (size)
        - روشنایی (brightness)
        - وضوح/تیزی (sharpness)
        - نسبت عرض/ارتفاع (aspect ratio)

        ورودی‌ها:
        - face_image: تصویر بریده‌شدهٔ صورت (BGR)
        - face_location: مکان صورت در فریم (برای کاربرد احتمالی)

        خروجی: عددی بین 0 و 1 نشان‌دهندهٔ کیفیت (در این پیاده‌سازی مقدار ثابتی برگردانده می‌شود)
        """
        try:
            # اندازه تصویر صورت (پیکسل)
            height, width = face_image.shape[:2]

            # نمرهٔ اندازه: نسبت اندازهٔ صورت به یک آستانهٔ 80x80، محدود شده به 1.0
            size_score = min(height * width / (80 * 80), 1.0)

            # تبدیل به خاکستری برای محاسبهٔ روشنایی و وضوح
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # روشنایی میانگین پیکسل‌ها (میان 0 تا 255)
            brightness = np.mean(gray)
            # نمرهٔ روشنایی: نزدیک بودن به مقدار 127 (میانه) بهتر است
            brightness_score = 1.0 - abs(brightness - 127) / 127

            # تیزی تصویر: واریانس لاپلاسیان (مقدار بالاتر یعنی تیزتر)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            # نرمال‌سازی نمرهٔ تیزی به بازهٔ [0,1] با یک عامل تقسیم
            sharpness_score = min(sharpness / 500, 1.0)

            # نسبت عرض به ارتفاع (برای بررسی اینکه برش خیلی کشیده یا پهن نباشد)
            aspect_ratio = width / height if height > 0 else 1
            # نمرهٔ نسبت: فاصله از نسبت مطلوب 0.8 کم‌تر باشد بهتر است
            aspect_score = 1.0 - min(abs(aspect_ratio - 0.8), 0.5)

            # ترکیب وزنی نمره‌ها (هر ویژگی وزنی دارد)
            actual_quality = (size_score * 0.3 + brightness_score * 0.2 +
                              sharpness_score * 0.3 + aspect_score * 0.2)

            # در کد اصلی مقدار ثابت 0.9 استفاده شده (برای تست) — در صورت نیاز می‌توان actual_quality را برگرداند
            # fixed_quality = 0.9
            # return actual_quality  # اگر می‌خواهید کیفیت واقعی را استفاده کنید
            return actual_quality

        except Exception as e:
            # در صورت بروز خطا به صورت محافظه‌کارانه مقدار 0.8 برمی‌گردانیم
            return 0.8

    def extract_and_cluster_faces(self, video_path, output_dir="faces", max_faces=50, frame_interval=10):
        """
        این تابع:
        1. ویدیو را باز می‌کند
        2. فریم‌ها را می‌خواند (هر frame_interval یکبار پردازش)
        3. چهره‌ها را تشخیص می‌دهد و encoding می‌گیرد
        4. اطلاعات چهره‌ها را جمع‌آوری می‌کند
        5. در پایان با فراخوانی cluster_faces_and_select_best خوشه‌بندی و انتخاب بهترین چهره را انجام می‌دهد

        پارامترها:
        - video_path: مسیر فایل ویدیو
        - output_dir: پوشه‌ای که عکس‌های منتخب در آن ذخیره می‌شوند
        - max_faces: حداکثر تعداد چهره‌ای که می‌خواهیم استخراج کنیم (محدودیت برای عملکرد)
        - frame_interval: هر چند فریم یک‌بار پردازش انجام شود
        """
        # ایجاد پوشه خروجی در صورت عدم وجود
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # بررسی وجود فایل ویدیو
        if not os.path.exists(video_path):
            print(f" Video not found: {video_path}")
            return []

        # باز کردن ویدیو با OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return []

        all_face_data = []  # لیست برای نگهداری اطلاعات تمام چهره‌های استخراج‌شده
        frame_count = 0
        print(f"🎥 Processing video: {video_path}")

        # حلقهٔ خواندن فریم‌ها
        while cap.isOpened() and len(all_face_data) < max_faces:
            ret, frame = cap.read()
            if not ret:
                # اگر فریم خوانده نشد => پایان ویدیو
                break

            # فقط هر frame_interval فریم پردازش می‌شود تا سرعت افزایش یابد
            if frame_count % frame_interval == 0:
                try:
                    # تبدیل BGR->RGB چون face_recognition از RGB استفاده می‌کند
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # تشخیص موقعیت چهره‌ها (لیست از (top, right, bottom, left))
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

                    if face_locations:
                        # استخراج encoding برای هر چهره
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        # پیمایش همزمان encoding و مکان‌ها
                        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):

                            # ایمن‌سازی محدوده‌ها تا از خروج از اندازهٔ تصویر جلوگیری شود
                            top, bottom = max(0, top), min(frame.shape[0], bottom)
                            left, right = max(0, left), min(frame.shape[1], right)

                            # اگر محدودهٔ برش معکوس بود رد می‌کنیم
                            if bottom <= top or right <= left:
                                continue

                            # برش تصویر صورت از فریم اصلی (BGR)
                            face_image = frame[top:bottom, left:right]

                            # نادیده گرفتن صورت‌های خیلی کوچک
                            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                                continue

                            # محاسبهٔ کیفیت برای انتخاب بهترین تصویر در خوشه
                            quality = self.calculate_face_quality(face_image, (top, right, bottom, left))

                            # ذخیرهٔ اطلاعات چهره در لیست
                            all_face_data.append({
                                'encoding': encoding,
                                'location': (top, right, bottom, left),
                                'image': face_image,
                                'quality_score': quality,
                                'frame_id': frame_count
                            })

                except Exception as e:
                    # در صورت خطا برای آن فریم فقط پیغام هشدار چاپ می‌شود و ادامه می‌دهیم
                    print(f"⚠️ Error in frame {frame_count}: {e}")

            # افزایش شمارندهٔ فریم
            frame_count += 1

        # آزادسازی منبع ویدیو
        cap.release()
        print(f"✅ {len(all_face_data)} faces extracted.")

        if not all_face_data:
            print("No faces found.")
            return []

        # فراخوانی خوشه‌بندی و انتخاب بهترین‌ها
        return self.cluster_faces_and_select_best(all_face_data, output_dir)

    def cluster_faces_and_select_best(self, all_face_data, output_dir):
        """
        1. خوشه‌بندی encodeها با DBSCAN و PCA
        2. برای هر خوشه (هر شخص) بهترین تصویر را بر اساس quality_score انتخاب می‌کند
        3. عکس‌های منتخب را ذخیره می‌کند و شناسهٔ یکتا تولید می‌نماید
        """
        # تبدیل لیست encodingها به آرایهٔ numpy برای محاسبات
        encodings = np.array([face['encoding'] for face in all_face_data])

        # 🔥 نمایش چهره‌های با کیفیت بالای 70% قبل از خوشه‌بندی
        high_quality_faces = [face for face in all_face_data if face['quality_score'] >= 0.7]
        print(f"🔹 Faces with quality >= 70%: {len(high_quality_faces)}/{len(all_face_data)}")
        
        # نمایش اطلاعات چهره‌های با کیفیت بالا
        for i, face in enumerate(high_quality_faces):
            print(f"   👤 High Quality Face {i+1}: Score = {face['quality_score']:.3f}, Frame = {face['frame_id']}")

        # اعمال PCA برای کاهش ابعاد
        if len(encodings) > 1:
            encodings_for_clustering = self.apply_pca(encodings)
            
            # محاسبهٔ پویا برای eps بر اساس داده‌های کاهش بعد یافته
            distances = pdist(encodings_for_clustering, 'euclidean')
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # تنظیم eps برای داده‌های PCA شده
            eps_value = min(0.6, max(0.4, mean_distance + 0.5 * std_distance))
        else:
            # اگر فقط یک چهره داریم، از PCA استفاده نکنیم
            encodings_for_clustering = encodings
            eps_value = 0.4

        print(f"🔹 Adaptive eps value = {eps_value:.3f}")

        # اجرای DBSCAN روی encodings کاهش بعد یافته
        clustering = DBSCAN(eps=eps_value, min_samples=1, metric='euclidean').fit(encodings_for_clustering)
        labels = clustering.labels_  # لیبل هر نمونه (هر برچسب یک خوشه است؛ -1 یعنی noise)

        # گروه‌بندی نمونه‌ها بر اساس برچسب
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(all_face_data[i])

        # تعداد خوشه‌های غیر-نویز
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"📊 Clustering result: {n_clusters} clusters, {list(labels).count(-1)} noise points")

        # اگر نمونه‌هایی با برچسب -1 (نویز) وجود داشتند: سعی می‌کنیم آن‌ها را به نزدیک‌ترین خوشه اختصاص دهیم
        noise_faces = clusters.pop(-1, [])
        for noise_face in noise_faces:
            best_match_id = None
            best_distance = 1.0 # حداکثر فاصله ممکن برای encodings در این پیاده‌سازی حدود 1.0 است

            # بررسی فاصله نویز به تمام خوشه‌های موجود و پیدا کردن کمترین فاصله
            for cluster_id, faces in clusters.items():
                cluster_encs = [f['encoding'] for f in faces]
                distances = face_recognition.face_distance(cluster_encs, noise_face['encoding'])
                min_dist = np.min(distances)

                if min_dist < best_distance:
                    best_distance = min_dist
                    best_match_id = cluster_id

            # اگر نویز به اندازهٔ کافی نزدیک به یک خوشه بود (آستانه 0.45) آن را اضافه می‌کنیم
            if best_distance < 0.45 and best_match_id is not None:
                clusters[best_match_id].append(noise_face)

        best_faces, best_encodings, best_ids = [], [], []

        # برای هر خوشه؛ بهترین تصویر را بر اساس quality_score انتخاب می‌کنیم
        for cluster_id, faces in clusters.items():
            # انتخاب عنصر با بیشترین نمرهٔ کیفیت
            best_face = max(faces, key=lambda x: x['quality_score'])

            # ساخت یک شناسهٔ یکتا برای هر فرد براساس شمارنده و نمرهٔ کیفیت
            face_id = f"person_{self.face_counter+1:03d}_q{best_face['quality_score']:.2f}"
            best_faces.append(best_face)
            best_encodings.append(best_face['encoding'])
            best_ids.append(face_id)

            # ذخیرهٔ عکس بهترین چهره برای آن خوشه
            cv2.imwrite(f"{output_dir}/{face_id}.jpg", best_face['image'])
            print(f"👤 Cluster {cluster_id}: {len(faces)} faces → Best Quality: {best_face['quality_score']:.3f}")
            self.face_counter += 1

        # 🔥 نمایش چهره‌های نهایی با کیفیت بالای 70%
        high_quality_final = [face for face in best_faces if face['quality_score'] >= 0.7]
        print(f"\n🎯 FINAL - High quality faces (>=70%): {len(high_quality_final)}/{len(best_faces)}")
        for i, face in enumerate(high_quality_final):
            print(f"   ✅ Person {i+1}: Quality = {face['quality_score']:.3f}")

        # به‌روزرسانی لیست‌های شناخته‌شده در شیء
        self.known_face_encodings = best_encodings
        self.known_face_ids = best_ids
        return best_faces

    def display_best_faces(self, faces_data, title="Unique Identified Faces"):
        """
        نمایش تصویری بهترین چهرهٔ هر فرد با matplotlib
        """
        if not faces_data:
            print("No faces to display.")
            return

        n = len(faces_data)
        cols = min(3, n)  # تعداد ستون‌ها در نما (حداکثر 3)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))
        plt.suptitle(f"{title}\nNumber of Unique Individuals: {n}", fontsize=16)

        for i, face_data in enumerate(faces_data):
            plt.subplot(rows, cols, i + 1)
            # تبدیل BGR->RGB برای نمایش صحیح در matplotlib
            face_rgb = cv2.cvtColor(face_data['image'], cv2.COLOR_BGR2RGB)
            plt.imshow(face_rgb)
            plt.title(f"Person {i+1}\nQuality: {face_data['quality_score']:.2f}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def display_high_quality_faces(self, faces_data, quality_threshold=0.7):
        """
        نمایش چهره‌های با کیفیت بالا
        """
        high_quality_faces = [face for face in faces_data if face['quality_score'] >= quality_threshold]
        
        if not high_quality_faces:
            print(f"🔹 No faces with quality >= {quality_threshold*100}%")
            return

        n = len(high_quality_faces)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))
        plt.suptitle(f"High Quality Faces (Quality >= {quality_threshold*100}%)\nTotal: {n} faces", fontsize=16)

        for i, face_data in enumerate(high_quality_faces):
            plt.subplot(rows, cols, i + 1)
            face_rgb = cv2.cvtColor(face_data['image'], cv2.COLOR_BGR2RGB)
            plt.imshow(face_rgb)
            plt.title(f"Quality: {face_data['quality_score']:.2f}\nFrame: {face_data['frame_id']}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# تابع اصلی برنامه که اجرا می‌شود
def main():
    comparator = EnhancedVideoFaceComparator()
    print("=" * 50)
    print("1=boy\n2=BOY2\n3=two girl")
    a = input("Choose video: ")

    video_map = {
        '1': "user_72959_video_5.mp4",
        '2': "user_93151_video_17.mp4",
        '3': "user_94842_video_0.mp4"
    }

    video_path = video_map.get(a)

    if not video_path or not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    print("🚀 Starting face extraction and clustering with PCA...")
    faces = comparator.extract_and_cluster_faces(
        video_path,
        output_dir="reference_faces",
        frame_interval=15,
        max_faces=50
    )

    if faces:
        print(f"\n🎉 Success! {len(faces)} unique individuals identified.\n")
        for i, f in enumerate(faces):
            print(f"Person {i+1}: Quality = {f['quality_score']:.2f}, Frame = {f['frame_id']}")
        
        # نمایش نتایج اصلی
        comparator.display_best_faces(faces)
        
        # 🔥 نمایش چهره‌های با کیفیت بالا
        print("\n" + "="*50)
        print("🔼 HIGH QUALITY FACES DISPLAY")
        comparator.display_high_quality_faces(faces, quality_threshold=0.7)
        
    else:
        print("No unique faces were identified.")


if __name__ == "__main__":
    main()