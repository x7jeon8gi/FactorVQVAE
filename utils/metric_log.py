import wandb

def log_metrics_as_bar_chart(metrics_dict):
    """
    metrics_dict: 예) {
        'IC': 0.12, 
        'IC_IR': 0.08,
        'RankIC': 0.10,
        'RankIC_IR': 0.07
    }
    
    이 딕셔너리 내용을 Table로 만들어 바 차트로 wandb에 로깅합니다.
    """
    
    # 1) 딕셔너리를 (Metric, Value) 형태의 2차원 리스트로 변환
    data = []
    for key, value in metrics_dict.items():
        # wandb가 float/int 등을 기대하므로, value가 numpy 타입이면 float()로 변환 권장
        data.append([key, float(value)])
    
    # 2) wandb.Table 객체 생성
    table = wandb.Table(data=data, columns=["Metric", "Value"])
    
    # 3) wandb.plot.bar를 통해 바 차트 생성
    bar_chart = wandb.plot.bar(
        table,      # wandb.Table 객체
        "Metric",   # x축 (카테고리)
        "Value",    # y축 (수치)
        title="Metrics Bar Chart"
    )
    
    # 4) 최종 wandb 로깅
    wandb.log({"my_bar_chart": bar_chart})