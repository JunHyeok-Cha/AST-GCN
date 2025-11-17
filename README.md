# AST-GCN

🚀 서울시 동적 교통량 예측 프로젝트: 데이터 산출물
‘서울시 동적 교통량 예측' (AST-GCN 모델) 프로젝트의 데이터 수집 및 정제 파트의 최종 산출물을 설명합니다.
이 폴더에는 모델 학습에 필요한 두 가지 핵심 파일이 포함되어 있습니다.
피처(Feature) 데이터 ($X$): data/FINAL_NODE_FEATURES_202510.csv
그래프(Graph) 데이터 ($A$): data/seoul_drive_net.gpkg

1. 📄 data/FINAL_NODE_FEATURES_202510.csv
이 파일은 AST-GCN 모델이 학습할 '피처(Feature) 행렬 $X$'입니다. 각 교차로(노드)가 시간대별로 얼마나 붐비는지를 나타내는 시계열 데이터입니다.
생성 과정:
서울시 버스 승하차 OpenAPI (API)
서울시 버스정류소 좌표 (Excel)
위 두 데이터를 결합하여 정류장별 시계열 피처 생성
'공간 결합'을 통해 각 정류장의 피처를 가장 가까운 '도로망 교차로 노드(osmid)'에 매핑
'교차로 노드(osmid)' 기준으로 시간대별 모든 피처를 합산(Group By)하여 생성
파일 형식: CSV (Comma-Separated Values)
컬럼 설명
컬럼명
설명
데이터 타입
예시
osmid
(Key) OpenStreetMap의 교차로 '노드' ID. (그래프 $A$의 노드와 일치)
Integer
282723804
Hour
(Time) 0시부터 23시까지의 시간대.
Integer
7
TotalTraffic
(Feature) 해당 노드/시간에 발생한 총 승하차 인원 (GetOn + GetOff).
Float
4202.0
GetOn
해당 노드/시간에 발생한 총 승차 인원.
Float
757.0
GetOff
해당 노드/시간에 발생한 총 하차 인원.
Float
3445.0
RouteCount
해당 노드에 매핑된 정류장들의 '경유 노선 수' 평균. (노드의 허브성)
Float
1.0


2. 🗺️ data/seoul_drive_net.gpkg
이 파일은 AST-GCN 모델의 '그래프(Graph) 뼈대 $A$'입니다. 서울시의 '차량용' 도로망(교차로, 도로) 구조를 담고 있는 지도 데이터 파일입니다.
생성 과정:
osmnx 라이브러리를 사용
OpenStreetMap(OSM) 서버에서 "Seoul, South Korea"의 "drive" 네트워크 다운로드
GeoPackage 파일로 저장
파일 형식: GeoPackage (.gpkg)
여는 방법:
코드: Python의 geopandas 라이브러리 (예: gpd.read_file('seoul_drive_net.gpkg', layer='nodes'))
프로그램: QGIS 등 무료 GIS 소프트웨어
파일 레이어 설명
이 파일 안에는 두 개의 레이어가 포함되어 있습니다.
nodes (노드/점):
그래프의 '노드(Node)'이며, 교차로 또는 분기점을 의미합니다.
FINAL_NODE_FEATURES...csv의 osmid 컬럼이 이 nodes 레이어의 osmid를 참조합니다.
edges (링크/선):
그래프의 '링크(Link)'이며, 노드와 노드를 연결하는 도로 구간을 의미합니다.
이 edges 정보를 바탕으로 모델이 학습할 '인접 행렬(Adjacency Matrix)'을 생성할 수 있습니다.



- 사용 모델
1. No-Graph Baseline (MLP / LSTM per node)
그래프 안 쓰고, 각 노드의 시간 시퀀스만 써서 예측
“그래프 정보를 쓰면 얼마나 좋아졌는지”를 보여줄 비교용

2. ST-GCN
Spatial: GCN (or ChebConv)
Temporal: 1D CNN / GRU
Attention, periodicity 없이 단순 구조

3. AST-GCN

“Baseline → ST-GCN → AST-GCN” 성능 향상 분석

+ 추후 발전된 모델 사용 가능