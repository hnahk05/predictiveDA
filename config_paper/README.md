# Config theo paper

Gom 4 dataset, moi dataset co pretraining.py va forecasting.py. Config goc trong configs/ duoc giu nguyen.

Thong so ap dung tu docs/paper.pdf (muc 4.2.3, 4.3.3):
- Transformer width: 64 o ca hai giai doan.
- Forecasting: batch 32; long_trend_hidden_dim va seasonality_hidden_dim deu 4; input 4032 buoc.
- Forecasting: 100 epochs, dien giai tu "100 training iterations" trong paper.

Cac gia dinh chua the xac nhan tu paper:
- Pretraining giu batch 8, 200 epochs va AdamW theo config goc vi paper khong tach rieng ngan sach pretraining.
- LR, milestones, gamma, curriculum learning, cac tham so khac va thiet lap CPU/GPU giu theo config goc; khong khang dinh la gia tri cua paper.
- Can pretrain lai checkpoint width 64 cho PEMS-BAY, PEMS04, PEMS08; checkpoint width 96 khong tuong thich.

Day la config khop cac thong so paper cong bo, khong phai cam ket tai hien chinh xac nhung thong so paper khong cong bo.

Cloud execution: config_paper uses one GPU for every stage/dataset. Pretraining loader/global batch=8; forecasting loader/global batch=32. PEMS-BAY pretraining GPU_NUM=8 inherited from the original config has been corrected to 1.
