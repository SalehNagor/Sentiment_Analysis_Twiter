import streamlit as st
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

# 1. إعداد الصفحة
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊")

st.title("📊 Sentiment Analysis System")
st.write("This dashboard uses a **Pruned DistilBERT** model to classify customer feedback.")

# 2. تحميل الموديل (يتم التحميل مرة واحدة فقط لتسريع الأداء)
@st.cache_resource
def load_model():
    # نحدد المسار: هل نحن داخل الدوكر أم على الجهاز المحلي؟
    # في الدوكر وضعناه في final_model، محلياً هو في models/pruned_model
    if os.path.exists("./final_model"):
        model_path = "./final_model"
    else:
        # المسار المحلي للموديل المخفف كما حددناه في main.py
        model_path = "./models/pruned_model" 
        
        # ملاحظة: إذا لم يجد التوكنايزر في مجلد pruned، قد تحتاج لتوجيهه لمجلد distilbert_finetuned
        # لكن الدوكر سيجمعهم، محلياً تأكد أن ملفات التوكنايزر موجودة بجانب الموديل
        if not os.path.exists(os.path.join(model_path, "vocab.txt")):
             # في حال كنت تشغل محلياً والتوكنايزر في مكان آخر
             tokenizer_path = "./models/distilbert_finetuned"
             return (DistilBertTokenizerFast.from_pretrained(tokenizer_path),
                     DistilBertForSequenceClassification.from_pretrained(model_path))

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# تحميل الموديل
try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully! ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. واجهة المستخدم للإدخال
text_input = st.text_area("Enter Customer Feedback:", height=100)

if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            # المعالجة (Tokenization)
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # التوقع (Prediction)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # تحويل النتائج لأرقام احتمالية
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

            # 4. ترجمة الأرقام إلى نصوص (حسب تدريبك: 0، 1، 2)
            # تأكد من ترتيب الكلاسات لديك، غالباً تكون كالتالي:
            labels_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😃"}
            sentiment = labels_map.get(predicted_class, "Unknown")

            # عرض النتيجة
            st.markdown("### Result:")
            if predicted_class == 2:
                st.success(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})")
            elif predicted_class == 0:
                st.error(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})")
            else:
                st.info(f"**Sentiment:** {sentiment} (Confidence: {confidence:.2%})")

st.write("Made by:")
st.write("Saleh Nagor / Majed Alfahmi / Anas Almuwalled / Rayan Aloufi")

st.write("Supervised by:")
st.write("Dr.Mohammed Arif")
