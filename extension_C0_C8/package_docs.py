"""Generate technical handoff from measured outputs, not academic prose."""
import json
from pathlib import Path
from pipeline import HERE,RUN

def main():
    s=json.loads((RUN/'status_final.json').read_text(encoding='utf-8'))
    lock=json.loads((RUN/'locked.json').read_text(encoding='utf-8'))
    pilot=json.loads((RUN/'pilot.json').read_text(encoding='utf-8'))
    lines=['# C0–C8: hồ sơ thực nghiệm residual head PEMS08','',
           'Đây là hướng dẫn code/bằng chứng, không phải báo cáo học thuật. Kết quả lịch sử trước lần tiếp tục được giữ trong `README_before_v2.md`, `status_before_v2.json`, `writing_guide_before_v2.md`.','',
           '| Bước | Trạng thái |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in s['status'].items()]
    lines += ['', '**C1_CONDITIONAL_PASS, vẫn NUMERIC_MISMATCH.** Không tuyên bố bitwise hay tương đương khoa học toàn bộ pipeline. Tiêu chí và giới hạn: `protocol_v2.json`, `run_v2/c1.json`. C0 PARTIAL vì chưa biết đầy đủ nguồn code preprocessing và tập dữ liệu fit pretrained. PASS ở các bước còn lại nghĩa đã thực thi/kiểm tra, không nghĩa mô hình luôn tốt hơn baseline.', '',
       '## Mô hình và dữ liệu', '',
       f"- Head chọn bằng validation nội vùng: `{s['selected_config']['id']}`, width {s['selected_config']['width']}; seed demo cố định 0 trước test.",
       '- Feature mỗi node gồm 12 prediction baseline chuẩn hóa, 12 Flow lịch sử ngắn chuẩn hóa, sin/cos giờ và ngày tương đối: 28 chiều. Không có target tương lai, Occupancy hay Speed trong feature head.',
       '- MLP dùng chung giữa node: Linear → ReLU → Linear; lớp cuối khởi tạo 0. Prediction gốc + 538,5 × head, alpha=1; alpha=0 chính xác baseline. Backbone đóng băng toàn bộ, eval, không AMP/TF32.',
       '- Giữ nguyên train/val/test 10700/3567/3566. Không đổi dataset, checkpoint epoch68 hoặc export gốc. Chọn head bằng nhãn validation nội vùng, bỏ 132 lượt nhãn chồng; vẫn báo metric đầy đủ riêng.',
       '- Baseline đã chọn checkpoint bằng validation gốc có nhãn chồng: hạn chế này còn tồn tại. Head học dự báo in-sample của backbone, không phải out-of-fold stacking.',
       '- Hai config h16/h32 seed0; sau khóa config chạy thêm seed1/2. Đây là 3 seed head trên cùng backbone, không phải 3 lần huấn luyện LSTTN.', '',
       '## Kết quả test đã đo', '',
       '| Phương pháp | Seed | Horizon phút | MAE | RMSE | MAPE % |','|---|---:|---:|---:|---:|---:|']
    for r in s['metrics']:
        lines.append(f"| {r['method']} | {r['seed']} | {r['horizon_minutes']} | {r['MAE']:.6f} | {r['RMSE']:.6f} | {r['MAPE_percent']:.6f} |")
    lines += ['', 'Nguồn: `run_v2/test_metrics.csv`. Chênh lệch dương/âm theo sensor/giờ/Flow: `run_v2/test_groups.csv`. Quartile trên target test chỉ để mô tả hậu nghiệm, không dùng tuning. Giờ là tương đối từ index; không có kết luận ngày thường/cuối tuần. Không có kiểm định ý nghĩa thống kê.', '',
       '## Cấu hình và tài nguyên', '',
       '| Config | Seed | Số tham số | Epoch chạy | MAE val nội vùng | Giây train+val | Peak VRAM GB | Peak RSS GB |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['head_runs']:
        lines.append(f"| {r['config']} | {r['seed']} | {r['parameters']} | {r['epochs']} | {r['validation_interior_MAE']:.6f} | {r['training_seconds']:.2f} | {r['peak_allocated_bytes']/1e9:.3f} | {r['peak_rss_bytes']/1e9:.3f} |")
    lines += ['', f"Cache nén thực tế: {s['cache_bytes']/1e6:.2f} MB. Nguồn số đo từng tập: `run_v2/cache/<split>/manifest.json`. Pilot batch2: {pilot['export_batch_seconds'][1:]}; không gán thời gian khởi động thành tốc độ ổn định.",
       'Theo dõi GPU: `run_v2/gpu_monitor.csv`; RSS lấy mẫu 10ms có thể bỏ lỡ đỉnh ngắn. Thời gian train từng run gồm validation; inference/cache/head có phạm vi đo khác nhau.', '',
       '| Phương án | Đánh giá |','|---|---|',
       '| RTX3050 local | Phương án đã dùng và ưu tiên: đủ bộ nhớ trong số đo thật, cache/checkpoint lưu local và resume được; không mất thời gian chuyển môi trường |',
       '| Kaggle | GPU/quota tài khoản và giới hạn phiên UNKNOWN; chưa đo |',
       '| Colab | GPU/quota/gói tài khoản/storage UNKNOWN; chưa đo |',
       '| GPU thuê | Nhà cung cấp, báo giá, lưu trữ, chi phí thực UNKNOWN; không thuê hoặc thanh toán |', '',
       '## XAI và phạm vi diễn giải', '',
       '- `run_v2/xai.json`, `xai_attributions.npz`, `xai_summary.csv`: 16 vị trí test đều theo index, 6 nhóm feature, 2 cách che, toàn bộ 170 sensor × 12 horizon.',
       '- Đây là attribution/sensitivity của **residual head**. Nhánh cộng baseline cố định; che prediction trong feature head không xóa baseline ở nhánh cộng. Không gọi là giải thích đầu vào thô của toàn LSTTN.',
       '- Che bằng trung bình train theo sensor hoặc 0 trong không gian feature. Flow normalized=0 là giữa thang MinMax, không phải Flow thô bằng0. Cặp sin/cos=(0,0) có thể ngoài manifold; hai giá trị che là kiểm tra độ nhạy, không là can thiệp nhân quả.',
       '- Không thay trọng số, không train lại architecture. IG thực hiện sau occlusion: xem `run_v2/ig_results.json`, `ig_summary.csv`, `ig_attributions.npz`, `ig_head_heatmap.png`. Chỉ giải thích head, kiểm tra completeness và độ ổn định 64/256 bước. Trạng thái IG: '+s.get('integrated_gradients',{}).get('status','NOT RUN')+'. Giá trị NOT RUN trong xai.json là tại thời điểm hoàn thành occlusion, trước bước IG bổ sung.', '',
       '## Chạy lại và resume', '',
       'Từ gốc repo, dùng venv hiện tại; các lệnh tái sử dụng artifact hoàn tất sau xác minh. Không chạy lại `verify` để cố đổi kết luận. Nếu thay config/code feature phải tạo phiên bản run riêng, không sửa protocol của run này.', '',
       '```powershell',
       '.\\venv\\Scripts\\python.exe -B extension_C0_C8\\test_pipeline.py',
       '.\\venv\\Scripts\\python.exe -B extension_C0_C8\\run_remaining.py',
       '.\\venv\\Scripts\\python.exe -B extension_C0_C8\\pipeline.py demo',
       '```', '',
       'Cache theo shard64, inference batch2; atomic rename, checksum và fingerprint. Head checkpoint resume mỗi epoch gồm optimizer, best weights và RNG shuffle; chạy lại bỏ qua run hoàn tất sau kiểm tra hash. Demo dùng hai mẫu cache, không chạy lại full test.', '',
       '## Bằng chứng kiểm thử', '',
       '- `run_v2/unit_tests_extended.log`: alpha0/zero init, gradient head, mask metric, feature không nhận target, Adam resume, loại nhãn chồng validation.',
       '- `run_v2/pilot.json`: inference/cache roundtrip, direct repeat, only-head gradients và tài nguyên.',
       '- `run_v2/final_integrity.json`: hash cache, index toàn bộ, direct-inference đối chiếu ở ba tập, hash baseline/export.',
       '- `run_v2/provenance.json`, `code_manifest.json`, `status_final.json`: config/hash/môi trường/trạng thái. Code mới chưa commit: không gán Git HEAD thành commit đã chứa thí nghiệm.',
       '- `run_v2/logs/`: log từng bước. `audit_gate.py` là gate lịch sử nghiêm ngặt trước v2, không dùng nó để phủ định quyết định conditional đã đăng ký.', '',
       '## Hình và hướng dẫn viết', '',
       'Các hình: `run_v2/test_MAE.png`, `test_RMSE.png`, `test_MAPE_percent.png`, `node_gain_60.png`, `head_learning_curves.png`, `xai_heatmap.png`, `xai_replacement_sensitivity.png` (đều dưới run_v2). Đọc `writing_guide.md` và tự điền `paper_reading.md`.']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    guide='''# Dàn ý tự viết và bảo vệ — dùng kết quả đã chạy

Không phải đoạn văn để nộp. Tự viết lập luận và trích dẫn; không suy từ PASS rằng mô hình cải thiện mọi metric.

## 1. Phương pháp đề xuất

- Tự giải thích công thức baseline + alpha × correction; phân biệt đơn vị normalized và Flow.
- Vẽ 28 feature → MLP nhỏ → 12 correction mỗi node; nhánh cộng baseline riêng.
- Nêu tại sao các feature biết được tại forecast origin; target chỉ dùng loss.
- Giải thích zero initialization, alpha0, backbone frozen/eval; dẫn code `pipeline.py:Head`, `Context.features`, `train_one`.
- Câu hỏi: head shared giữa sensor có học quan hệ trực tiếp giữa sensor không? Vì sao không thêm graph fix cùng thí nghiệm?

## 2. Thiết lập thực nghiệm

- Dùng `protocol_v2.json` để tự mô tả hai config, Adam/lr/epoch/patience, metric chọn val, seed.
- Dùng `run_v2/provenance.json` và `code_manifest.json` cho hash/config/môi trường.
- Nêu split thật 60/20/20 và nhãn chồng. Giải thích mask validation nội vùng loại 132 lượt nhãn, không đổi split hay metric test.
- Nêu checkpoint backbone có lịch sử chọn val cũ; pretraining data scope vẫn UNKNOWN.
- Câu hỏi: vì sao head 3 seed không phải LSTTN 3 seed? Vì sao train prediction không out-of-fold là hạn chế?

## 3. Bảng và hình kết quả cần dùng

| Bảng/hình | Nguồn file dưới extension_C0_C8/run_v2 | Điều cần tự phân tích |
|---|---|---|
| C1 numerical vs metric | c1.json, c1_numeric_violations.csv, prior_cpu_gpu_violations.csv | Vì sao conditional pass không phải bitwise pass? |
| Hai config và các seed thực chạy | config_seed_results.csv; heads/*/result.json | Config chọn bằng metric nào, có train đủ12epoch không? |
| Mean/std validation seed | validation_summary.json | Std giữa seed head có giới hạn gì? |
| MAE/RMSE/MAPE 15/30/60 | test_metrics.csv; test_MAE.png, test_RMSE.png, test_MAPE_percent.png | Metric/horizon nào tăng, giảm? Không bỏ hàng bất lợi |
| Gain/loss node/giờ/Flow | test_groups.csv; node_gain_60.png | Có node suy giảm không? Quartile hậu nghiệm dùng để làm gì? |
| Đường học | head_learning_curves.png; heads/*/result.json | Điểm chọn checkpoint, khả năng overfit |
| Tài nguyên | pilot.json; cache/*/manifest.json; heads/*/result.json; gpu_monitor.csv | Tách khởi động, cache, train, RAM và VRAM |
| XAI | xai_summary.csv; xai_heatmap.png; xai_replacement_sensitivity.png | Kết luận có đổi khi đổi giá trị che không? |

## 4. XAI

- Nêu scope head, sample selection cố định 16 vị trí, sensor/horizon, các nhóm feature, hai cách che.
- Raw attribution có trong `xai_attributions.npz`; index và cấu hình trong `xai.json`.
- Câu hỏi: vì sao che cached prediction vẫn giữ nhánh cộng baseline? Có được nói đây là đóng góp lịch sử thô vào toàn LSTTN không?
- Câu hỏi: Flow normalized0 có nghĩa gì? Cặp sin/cos0 có hợp lệ không? Vì sao không suy nhân quả?
- IG bổ sung: `ig_results.json`, `ig_summary.csv`, `ig_attributions.npz`, `ig_head_heatmap.png`. Tự kiểm tra trạng thái PASS/PARTIAL, completeness và độ ổn định 64/256 bước; không gộp với occlusion hoặc nói đây là IG của backbone.

## 5. Hạn chế và hướng phát triển

- Tự đánh giá numeric mismatch, provenance chưa đủ, overlap từ baseline, calendar chưa xác minh.
- In-sample residual learning, một backbone seed, ít cấu hình và không có kiểm định ý nghĩa thống kê.
- XAI chỉ head và nhạy với reference; không gọi là ablation huấn luyện lại.
- Mọi thí nghiệm mới chưa chạy ghi [CHỜ KẾT QUẢ]; không dùng test để chỉnh lại cấu hình sau khi xem bảng.

## 6. Việc bạn tự làm

1. Đọc bốn paper và tự điền phiếu trong `paper_reading.md`, kiểm tra trang/phương trình và nguồn trích dẫn.
2. Tự viết theo mẫu trường, Việt hóa bảng/hình/công thức và kiểm tra tất cả số liệu với CSV.
3. Giải thích được một cửa sổ dữ liệu, một batch head, metric masking và việc không truyền gradient vào backbone.
4. Tập demo `pipeline.py demo`; metric demo2mẫu không phải metric toàn test.
5. Tự commit code và bằng chứng cần thiết lên GitHub theo yêu cầu môn; cân nhắc kích thước cache/checkpoint, không commit dữ liệu lớn vô ý. Agent chưa push hoặc tạo commit.
'''
    (HERE/'writing_guide.md').write_text(guide,encoding='utf-8')
    print('Technical handoff updated from measured results')

if __name__=='__main__':main()
