# AST-GCN

### 데이터
서울시 동적 교통량 예측 프로젝트: 데이터 산출물
‘서울시 동적 교통량 예측' (AST-GCN 모델) 프로젝트의 데이터 수집 및 정제 파트의 최종 산출물을 설명합니다.
이 폴더에는 모델 학습에 필요한 두 가지 핵심 파일이 포함되어 있습니다.
피처(Feature) 데이터 ($X$): res/SEOUL_TRAFFIC_DATASET_2025.csv (1~10월)
그래프(Graph) 데이터 ($A$): res/seoul_drive_net.gpkg

1. SEOUL_TRAFFIC_DATASET_2025.csv (Feature Matrix)
서울시 도로 교차로(Node)별 시간대별 교통 상황을 담은 시계열 데이터입니다. **버스 승하차 인원(수요)**과 **실제 차량 속도(혼잡도)**가 하나의 테이블로 통합
    - 데이터 기간: 2025년 1~10월 (10개월)
    - 시간 단위: 1시간 (0시 ~ 23시)
    - 공간 단위: OpenStreetMap(OSM) 노드 ID (osmid)
버스 데이터: 서울시 버스 승하차 데이터를 Spatial Join을 통해 가장 가까운 교차로 노드에 매핑.
속도 데이터: 서울시 도로별 통행속도 데이터를 도로명(Road Name)을 기준으로 OSM 링크와 매칭하여 노드에 할당.



2. 🗺️ data/seoul_drive_net.gpkg
이 파일은 AST-GCN 모델의 '그래프(Graph) 뼈대입니다. 서울시의 '차량용' 도로망(교차로, 도로) 구조를 담고 있는 지도 데이터 파일입니다.
생성 과정:
    osmnx 라이브러리를 사용
    OpenStreetMap(OSM) 서버에서 "Seoul, South Korea"의 "drive" 네트워크 다운로드
    GeoPackage 파일로 저장
파일 형식:
    GeoPackage (.gpkg)
여는 방법:
    코드: Python의 geopandas 라이브러리 (예: gpd.read_file('seoul_drive_net.gpkg', layer='nodes'))
    프로그램: QGIS 등 무료 GIS 소프트웨어
파일 레이어:
이 파일 안에는 두 개의 레이어가 포함되어 있습니다.
    nodes (노드/점):
        그래프의 '노드(Node)'이며, 교차로 또는 분기점을 의미합니다.SEOUL_TRAFFIC_DATASET_2025.csv의 osmid 컬럼이 이 nodes 레이어의 osmid를 참조합니다.
    edges (링크/선):
        그래프의 '링크(Link)'이며, 노드와 노드를 연결하는 도로 구간을 의미합니다.이 edges 정보를 바탕으로 모델이 학습할 '인접 행렬(Adjacency Matrix)'을 생성할 수 있습니다.



### 사용 모델
1. No-Graph Baseline (MLP / LSTM per node)
그래프 안 쓰고, 각 노드의 시간 시퀀스만 써서 예측
“그래프 정보를 쓰면 얼마나 좋아졌는지”를 보여줄 비교용
    - 버스 승하차 정보를 입력, 버스 승하차 정보를 출력
    - 버스 승하차 정보 + 차량 속도를 입력, 버스 승하차 정보를 출력
    - 버스 승하차 정보 + 차량 속도를 입력, 버스 승하차 정보 + 차량 속도를 출력
세 경우로 나누어서 테스트 =>
- 속도 feature가 모델 성능 향상에 도움이 되는지 확인
- 버스 승하차 정보와 속도를 동시에 예측하는 것이 단일 목표(승하차량만 예측) 대비 추가적인 성능 향상과 실용적 이점을 제공하는지 평가

2. ST-GCN
정적인 도로 네트워크 그래프를 사용해 인접 노드 간 공간 상관성을 반영하는 기본 GNN 모델
Spatial: GCN (or ChebConv) 기반 합성곱으로 인접 노드 정보 집계
Temporal: 1D CNN / GRU로 시간 축 패턴 학습
    - Attention, Daily/Weekly 주기 모듈 없이 최근 구간(Recent window)만 사용하는 단순 구조
=> No-Graph Baseline과 비교하여 “그래프 구조만 추가했을 때 얻을 수 있는 성능 이득”을 평가

3. AST-GCN
중요한 노드와 시간대에 시공간 Attention을 부여하여 동적으로 교통량 예측하는 모델
    - 가중치 부여, only Recent window만
    - 가중치 부여, Recent, Daily, Weekly window로 각각 모델 돌린 후 퓨전(periodicity Modeling)
두 경우로 나누어서 테스트 =>
- 단순 그래프 기반 ST-GCN 대비 Attention 자체가 얼마나 성능 향상에 기여하는지 평가
- 일/주 단위 주기 정보를 함께 사용할 때 추가적인 성능 향상이 발생하는지 평가

“Baseline → ST-GCN → AST-GCN” 순으로 그래프 정보, Attention, 주기 정보가 단계적으로 추가되면서 성능이 향상되는지를 분석한다.

+ 추후 발전된 모델 사용 가능


### 분석
모든 분석은 5개의 모델별 산출 결과 평균으로 비교한다.

1-1. No-Graph Baseline (MLP)
    - 버스 승하차 정보를 입력, 버스 승하차 정보를 출력
        [Test] Loss: 0.0630, MAE: 0.1127
    - 버스 승하차 정보 + 차량 속도를 입력, 버스 승하차 정보를 출력
        [Test] Loss: 0.0635, MAE: 0.1090
    - 버스 승하차 정보 + 차량 속도를 입력, 버스 승하차 정보 + 차량 속도를 출력
        [Test] Loss: 0.0373, MAE(all): 0.0869, MAE(traffic): 0.1169, MAE(speed): 0.0570

1-2. No-Graph Baseline (LSTM)
    - 버스 승하차 정보를 입력, 버스 승하차 정보를 출력
        [Test] Loss: 0.0577, MAE: 0.1067
    - 버스 승하차 정보 + 차량 속도를 입력, 버스 승하차 정보를 출력
        [Test] Loss: 0.0455, MAE: 0.0941
    - 버스 승하차 정보 + 차량 속도를 입력, 버스 승하차 정보 + 차량 속도를 출력
        [Test] Loss: 0.0322, MAE(all): 0.0797, MAE(traffic): 0.1074, MAE(speed): 0.0518


(1) 속도 feature 입력 효과
    - MLP
    승하차 → 승하차: MAE 0.1127
    (승하차+속도) → 승하차: MAE 0.1090
        속도를 같이 넣어주면 MAE가 약간 감소 → 속도 정보가 승하차량 예측에 어느 정도 도움이 됨.

    - LSTM
    승하차 → 승하차: MAE 0.1067
    (승하차+속도) → 승하차: MAE 0.0941
        MLP보다 더 크게 개선됨 → 시계열 구조를 갖는 LSTM은 속도 feature를 잘 활용해서 승하차 패턴을 더 잘 잡는 편이라고 해석 가능.

=> “그래프 없이도” 속도 feature를 추가하는 것만으로 승하차 예측 성능이 꾸준히 좋아지며, 특히 LSTM이 이 이득을 더 크게 가져간다.

(2) 다중 타깃 예측 (승하차 + 속도 동시 예측)

    - MLP
    (승하차+속도) → (승하차+속도)
    MAE(all) 0.0869
    MAE(traffic): 0.1169 (단일 타깃 0.1090보다 악화)
    MAE(speed): 0.0570

=> 전체 평균 MAE는 낮지만, 승하차 자체의 오차는 오히려 조금 증가.
멀티태스크 학습이 속도 쪽엔 유리하지만, 승하차만 보았을 땐 단일 타깃보다 불리할 수 있음.

    - LSTM
    (승하차+속도) → (승하차+속도)
    MAE(all) 0.0797
    MAE(traffic): 0.1074 (단일 타깃 0.0941보다 약간 악화)
    MAE(speed): 0.0518

=> MLP와 비슷하게, 전체 평균 관점(MAE(all))에서는 가장 좋지만,
승하차만 놓고 보면 (승하차+속도) → 승하차 단일 타깃 모델이 제일 좋음.

결론: 다중 타깃 예측은 “전체적인 평균 성능”을 낮추는 데는 효과적이지만, 우리가 관심 있는 승하차 예측 정확도는 단일 타깃 모델이 더 좋을 수 있음.
즉, “실용적 관점(속도까지 같이 예측해 활용)” vs “순수 승하차 정확도” 사이 trade off가 존재함.



2. ST-GCN
    - Attention, Daily/Weekly 주기 모듈 없이 최근 구간(Recent window)만 사용하는 단순 구조
        [Test] Loss: 0.0422, MAE(all): 0.0934, MAE(traffic): 0.1292, MAE(speed): 0.0576

전체 평균 MAE(all) 기준
    MAE(all): ST-GCN(0.0934) < No-Graph LSTM(0.0797)
    MAE(traffic): ST-GCN(0.1292) < No-Graph LSTM(0.1074)
    MAE(speed): ST-GCN(0.0576) < No-Graph LSTM(0.0518)
=> 그래프 구조를 쓴다고 해서 자동으로 전체 성능이 좋아지지는 않음.

결론: 정적인 도로 네트워크 그래프를 반영하는 ST-GCN은, 이번 실험 설정에서는 No-Graph LSTM 대비 전체 MAE 기준으로 뚜렷한 우위를 보이지 못했다.
이는 서울 시내 버스 노드 간 그래프 구조가 현재 설계된 형태의 ST-GCN에 충분히 활용되지 못했거나, 단순한 LSTM이 이미 시계열 패턴을 잘 포착하고 있기 때문으로 해석할 수 있다.


3. AST-GCN
    - 가중치 부여, only Recent window만
        [Test] Loss: 0.3163, MAE(all): 0.2974, MAE(traffic): 0.1519, MAE(speed): 0.4429
    - 가중치 부여, Recent, Daily, Weekly window로 각각 모델 돌린 후 퓨전(periodicity Modeling)
        [Test] Loss: 0.0492, MAE(all): 0.1002, MAE(traffic): 0.0936, MAE(speed): 0.1068

(1) Only Recent + Attention
    - AST-GCN (Attention, only Recent)
=> 모든 지표가 명백하게 나쁨
학습이 안정적으로 수렴하지 못해 불안정하거나, 하이퍼파라미터/구현 문제로 Attention이 오히려 발산/과적합을 유도한 실패 케이스에 가깝다.

(2) Recent + Daily + Weekly + Attention (Periodicity Fusion)
    - AST-GCN (periodicity modeling)
=> ST-GCN + No-Graph와 비교하면

    - AST-GCN vs ST-GCN
    traffic: 0.0936 (AST) < 0.1292 (ST) → 승하차 예측은 확실히 개선
    speed: 0.1068 (AST) > 0.0576 (ST) → 속도 예측은 오히려 악화
    all: 0.1002 (AST) > 0.0934 (ST) → 전체 평균은 ST-GCN이 더 좋음

    - AST-GCN vs No-Graph LSTM (다중 타깃)
    traffic: 0.0936 (AST) < 0.1074 (LSTM) → 승하차 예측은 AST-GCN이 가장 좋음
    speed: 0.1068 (AST) > 0.0518 (LSTM) → 속도 예측은 가장 나쁨
    all: 0.1002 (AST) > 0.0797 (LSTM)

결과: 버스 승하차 예측만 보면 AST-GCN(Periodicity)이 모든 모델 중 가장 작은 MAE(0.0936)
→ 일/주 단위 주기 정보를 Attention과 함께 활용하는 것이 승하차량 패턴을 잡는 데 효과적.
반대로, 속도 예측은 AST-GCN에서 크게 나빠져 전체 MAE도 악화.
-> 주기성/Attention이 승하차에는 도움이 되지만, 속도에는 잡음 또는 과도한 복잡성으로 작용했을 가능성.

즉, “우리가 어떤 타깃에 더 관심이 있는지”에 따라 모델 선택 결론이 달라짐:
승하차량 예측이 핵심이라면: AST-GCN(periodicity)가 의미 있는 개선.
승하차 + 속도 전체 평균을 최소화하려면: 여전히 No-Graph LSTM 멀티타깃이 강력.

+ 앞으로 해야할 점
속도 예측을 별도 모듈로 분리하거나, 멀티태스크 손실 가중치를 조정하는 방식으로 승하차·속도 간 트레이드오프를 제어하는 확장이 필요하다.