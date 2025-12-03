import sys
import os
# إضافة المجلد الرئيسي للمسار لكي نتمكن من استيراد المديولات
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_text

def test_clean_text_basic():
    """اختبار تنظيف النصوص الأساسي"""
    raw_text = "Hello <br> World! 123"
    cleaned = clean_text(raw_text)
    # نتوقع أن يحذف الـ HTML والأرقام ويحول الحروف لصغيرة
    assert cleaned == "hello  world " or cleaned == "hello world"

def test_clean_text_no_change():
    """اختبار نص نظيف أصلاً"""
    text = "pure text"
    assert clean_text(text) == "pure text"