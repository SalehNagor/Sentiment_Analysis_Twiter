# استخدام صورة بايثون خفيفة
FROM python:3.9-slim

# إعداد متغيرات البيئة لمنع ملفات pyc ولجعل الـ log يظهر فوراً
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تحديد مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملف المتطلبات وتثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- الجزئية المهمة: تجميع المودل الـ Pruned ---

# 1. إنشاء مجلد داخل الدوكر لاستقبال المودل النهائي
RUN mkdir -p /app/final_model

# 2. نسخ الـ Tokenizer من مجلد التدريب الأول (لأننا لم نحفظه مع الـ Pruned)
# ننسخ الملفات الأساسية للـ Tokenizer
COPY models/distilbert_finetuned/vocab.txt /app/final_model/
COPY models/distilbert_finetuned/tokenizer_config.json /app/final_model/
COPY models/distilbert_finetuned/special_tokens_map.json /app/final_model/
# (احتياطاً) لو وُجد ملف tokenizer.json انسخه أيضاً، أو تجاهل هذا السطر إذا لم يوجد
COPY models/distilbert_finetuned/tokenizer.json /app/final_model/ 2>/dev/null || :

# 3. نسخ أوزان المودل المخفف (Pruned Weights) من المجلد الثاني
# هذا سينسخ config.json و pytorch_model.bin الخاصين بالمودل المخفف
COPY models/pruned_model/* /app/final_model/

# ------------------------------------------------

# نسخ كود التطبيق (app.py) وباقي ملفات المشروع
COPY . .

# فتح المنفذ (اختياري حسب السيرفر)
EXPOSE 8501

# أمر التشغيل (تأكد أن كود app.py يقرأ المودل من المسار الجديد)
CMD ["streamlit", "run", "app.py"]