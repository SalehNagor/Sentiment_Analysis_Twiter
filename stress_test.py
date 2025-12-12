import time
import requests
import concurrent.futures

# 1. التعديل المهم: الرابط يجب أن يكون لرابط التوقع، والنوع POST
API_URL = "http://localhost:8000/predict" 
# تأكد أن المنفذ 8000 (FastAPI) وليس 8501 (Streamlit) لأننا نختبر الـ API مباشرة

# بيانات حقيقية لكي يقوم المودل بمعالجتها فعلاً
TEST_PAYLOAD = {"text": "I really love this service, it works great and fast!"}

def send_request(request_id):
    start_time = time.time()
    try:
        # نستخدم POST بدلاً من GET لاختبار المودل
        response = requests.post(API_URL, json=TEST_PAYLOAD, timeout=10)
        
        # نعتبره نجاحاً فقط إذا رد المودل بنتيجة (200 OK)
        if response.status_code == 200:
            status = "Success"
        else:
            status = "Failed"
    except Exception as e:
        status = "Connection Error"
    
    end_time = time.time()
    return status, end_time - start_time

def run_stress_test(total_requests=100, concurrent_users=20):
    print(f"\n--- 🚀 Starting Stress Test (Testing Model Inference) ---")
    print(f"Target: {API_URL} | Users: {concurrent_users} | Reqs: {total_requests}")
    
    start_all = time.time()
    
    # تنفيذ الهجوم
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = list(executor.map(send_request, range(total_requests)))

    end_all = time.time()
    total_time = end_all - start_all

    # حساب النتائج
    success_count = sum(1 for r in results if r[0] == "Success")
    failed_count = total_requests - success_count
    avg_time = sum(r[1] for r in results) / len(results)
    throughput = total_requests / total_time

    # --- طباعة التقرير (هذا ما يجب تصويره) ---
    print("\n" + "="*45)
    print("      📊 LOAD & STRESS TEST REPORT      ")
    print("="*45)
    print(f"✅ Total Requests:       {total_requests}")
    print(f"👥 Concurrent Users:     {concurrent_users}")
    print(f"🟢 Successful Requests:  {success_count}")
    print(f"🔴 Failed Requests:      {failed_count}")
    print("-" * 45)
    print(f"⏱️ Average Latency:      {avg_time:.4f} seconds")
    print(f"⚡ System Throughput:    {throughput:.2f} reqs/sec")
    print(f"🕒 Total Duration:       {total_time:.2f} seconds")
    print("="*45 + "\n")

if __name__ == "__main__":
    try:
        run_stress_test()
    except Exception as e:
        print(f"Error: {e}. Ensure 'uvicorn src.api:app' is running on port 8000.")