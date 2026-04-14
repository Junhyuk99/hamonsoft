"""
SFR-002: 장애 발생 전 징후 포착 대시보드
Streamlit 기반 실시간 이상 탐지 시각화

실행 방법:
    streamlit run app_anomaly_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import os

# 페이지 설정
st.set_page_config(
    page_title="NETIS 이상 탐지 시스템",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 10px;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
    }
    .alert-normal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============== 데이터 로딩 ==============
@st.cache_data
def load_data():
    """데이터 로딩 및 전처리"""
    DATA_PATH = './output/'

    data = {}

    try:
        # NMS 데이터
        data['nms_master'] = pd.read_csv(f'{DATA_PATH}01_NMS_장비마스터.csv', encoding='utf-8-sig')
        data['nms_if_perf'] = pd.read_csv(f'{DATA_PATH}04_NMS_IF성능_5min.csv', encoding='utf-8-sig')

        # SMS 데이터
        data['sms_master'] = pd.read_csv(f'{DATA_PATH}08_SMS_장비마스터.csv', encoding='utf-8-sig')
        data['sms_cpu'] = pd.read_csv(f'{DATA_PATH}10_SMS_CPU_5min.csv', encoding='utf-8-sig')
        data['sms_memory'] = pd.read_csv(f'{DATA_PATH}11_SMS_메모리_5min.csv', encoding='utf-8-sig')
        data['sms_filesystem'] = pd.read_csv(f'{DATA_PATH}12_SMS_파일시스템_5min.csv', encoding='utf-8-sig')

        # 시간 변환
        for key in ['nms_if_perf', 'sms_cpu', 'sms_memory', 'sms_filesystem']:
            if key in data and 'YMDHMS' in data[key].columns:
                data[key]['DATETIME'] = pd.to_datetime(
                    data[key]['YMDHMS'].astype(str),
                    format='%Y%m%d%H%M%S',
                    errors='coerce'
                )

        # CPU 사용률 계산
        if 'sms_cpu' in data and 'IDLE_PCT_AVG' in data['sms_cpu'].columns:
            data['sms_cpu']['CPU_USAGE_AVG'] = 100 - data['sms_cpu']['IDLE_PCT_AVG']

        return data

    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return None


# ============== 이상 탐지 클래스 ==============
class AnomalyDetector:
    """통합 이상 탐지 클래스"""

    @staticmethod
    def detect_zscore(series, threshold=3.0):
        """Z-Score 기반 이상 탐지"""
        z_scores = np.abs(stats.zscore(series.fillna(series.median())))
        return z_scores > threshold

    @staticmethod
    def detect_iqr(series, multiplier=1.5):
        """IQR 기반 이상 탐지"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - multiplier * IQR) | (series > Q3 + multiplier * IQR)

    @staticmethod
    def detect_threshold(series, warning=80, critical=90):
        """임계값 기반 이상 탐지"""
        severity = pd.Series('normal', index=series.index)
        severity[series >= warning] = 'warning'
        severity[series >= critical] = 'critical'
        return severity

    @staticmethod
    def detect_isolation_forest(df, feature_cols, contamination=0.05):
        """Isolation Forest 기반 이상 탐지"""
        features = df[feature_cols].fillna(df[feature_cols].median())
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = model.fit_predict(features_scaled)

        return predictions == -1, model.decision_function(features_scaled)


# ============== LLM 분석기 (내부망 Ollama) ==============
import time

class LLMAnalyzer:
    """
    내부망 Ollama 서버를 이용한 LLM 분석기

    서버 정보:
    - HOST: http://210.107.60.21:11434
    - MODEL: gpt-oss:120b (SKT A.X-4.0-Light 기반)
    """

    # 내부망 서버 설정
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://210.107.60.21:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
    RETRIES = 3
    TEMPERATURE = 0.7

    def __init__(self, base_url=None, model=None):
        self.base_url = (base_url or self.OLLAMA_HOST).rstrip("/")
        self.model = model or self.OLLAMA_MODEL

    def check_connection(self):
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False

    def analyze(self, prompt, temperature=None):
        """
        LLM에 분석 요청 (재시도 로직 포함)
        - /api/chat 우선 시도
        - 실패시 /api/generate로 fallback
        """
        system_prompt = """당신은 IT 인프라 전문가입니다.
네트워크 및 서버 모니터링 데이터를 분석하고, 이상 징후의 원인과 영향을 파악하며,
적절한 대응 방안을 제시합니다. 한국어로 답변해주세요."""

        temp = temperature or self.TEMPERATURE
        last_err = None

        # 1차 시도: /api/chat 엔드포인트
        payload_chat = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": temp,
                "num_ctx": 8192,
            },
            "stream": False,
        }

        for attempt in range(1, self.RETRIES + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload_chat,
                    timeout=180
                )
                r.raise_for_status()
                data = r.json()
                result = data.get("message", {}).get("content", "").strip()
                if result:
                    return result
            except Exception as e:
                last_err = e
                time.sleep(0.4 * attempt)

        # 2차 시도: /api/generate 엔드포인트 (fallback)
        formatted_prompt = f"### 시스템 지시 ###\n{system_prompt}\n\n### 사용자 입력 ###\n{prompt}\n"
        payload_gen = {
            "model": self.model,
            "prompt": formatted_prompt,
            "options": {
                "temperature": temp,
                "num_ctx": 8192,
            },
            "stream": False,
        }

        for attempt in range(1, self.RETRIES + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload_gen,
                    timeout=180
                )
                r.raise_for_status()
                data = r.json()
                result = data.get("response", "").strip()
                if result:
                    return result
            except Exception as e:
                last_err = e
                time.sleep(0.4 * attempt)

        return f"분석 실패: {last_err}"


# ============== 원본 데이터 EDA 페이지 ==============
def parse_column_descriptions():
    """컬럼 설명 파일 파싱"""
    column_desc = {}

    def parse_column_line(line):
        """컬럼 정보 라인 파싱 (고정 너비 형식)
        예: '  1  ENG_NO                         decimal(5,0)              YES         엔진번호'
        위치: 1-5:번호, 6-36:컬럼명, 37-61:타입, 62-66:NULL, 67-71:Key, 72+:코멘트
        """
        line = line.replace('\r', '')  # 캐리지 리턴 제거

        # 최소 길이 체크
        if len(line) < 40:
            return None, None

        try:
            # 고정 너비 파싱
            col_name = line[5:36].strip()
            comment = line[71:].strip() if len(line) > 71 else ''

            if col_name and not col_name.startswith('-'):
                return col_name, comment
        except:
            pass

        return None, None

    # output 폴더 - 00_컬럼설명.txt
    try:
        with open('./output/00_컬럼설명.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_table = None

        for line in lines:
            line = line.replace('\r', '').rstrip('\n')

            # 파일명 추출: "01_NMS_장비마스터.csv  (원본: cm_Dev10)"
            if '.csv' in line and '(원본:' in line:
                table_name = line.split('.csv')[0].strip()
                current_table = table_name
                column_desc[current_table] = {}
            # 컬럼 정보 라인 (숫자로 시작)
            elif current_table and line.strip() and len(line) > 5:
                first_part = line[:5].strip()
                if first_part.isdigit():
                    col_name, comment = parse_column_line(line)
                    if col_name:
                        column_desc[current_table][col_name] = comment
    except Exception as e:
        pass

    # dbtable 폴더 - _테이블설명.txt
    try:
        with open('./dbtable/_테이블설명.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_table = None

        for line in lines:
            line = line.replace('\r', '').rstrip('\n')

            # 테이블명 추출: "cm_Dev10  (457건, 78컬럼)"
            if '(' in line and '건' in line and '컬럼' in line and '==' not in line:
                table_name = line.split('(')[0].strip()
                current_table = table_name
                column_desc[current_table] = {}
            # 컬럼 정보 라인
            elif current_table and line.strip() and len(line) > 5:
                first_part = line[:5].strip()
                if first_part.isdigit():
                    col_name, comment = parse_column_line(line)
                    if col_name:
                        column_desc[current_table][col_name] = comment
    except Exception as e:
        pass

    return column_desc


# ============== 이상치 탐지 활용 가이드 페이지 ==============
def page_anomaly_guide():
    """이상치 탐지 활용 가이드 페이지"""
    st.title("🎯 이상치 탐지 활용 가이드")
    st.markdown("**각 데이터 열별 이상치 탐지 알고리즘 활용 방안**")

    # 컬럼 설명 로드
    column_descriptions = parse_column_descriptions()

    # ===== 1. SMS 서버 모니터링 =====
    st.header("🖥️ SMS 서버 모니터링")

    st.subheader("1️⃣ CPU 이상 탐지")
    cpu_data = {
        '열': ['CPU_USAGE_AVG (100-IDLE_PCT_AVG)', 'USER_PCT_AVG', 'SYSTEM_PCT_AVG', 'IOWAIT_PCT_AVG', 'RUN_QUEUE_AVG'],
        '설명': ['전체 CPU 사용률', '사용자 프로세스 CPU', '시스템 커널 CPU', 'I/O 대기 비율', '실행 대기 프로세스 수'],
        '적합 알고리즘': ['Z-Score, 임계값(80/90%)', 'Z-Score', 'Z-Score', 'Z-Score (I/O 병목)', 'Z-Score (과부하)'],
        '이상 시나리오': ['서버 과부하', '애플리케이션 부하', '커널 이슈', '디스크 병목', 'CPU 포화'],
        '파일': ['10_SMS_CPU_5min.csv'] * 5
    }
    st.dataframe(pd.DataFrame(cpu_data), width='stretch')

    st.subheader("2️⃣ 메모리 이상 탐지")
    mem_data = {
        '열': ['PHYSICAL_USED_PCT', 'SWAP_USED_PCT', 'BUFFER_SIZE', 'CACHE_SIZE'],
        '설명': ['물리 메모리 사용률', '스왑 메모리 사용률', '버퍼 크기', '캐시 크기'],
        '적합 알고리즘': ['Z-Score, 임계값(80/90%)', 'Z-Score (스왑 사용 = 메모리 부족)', 'IQR', 'IQR'],
        '이상 시나리오': ['메모리 부족', '물리 메모리 고갈', '버퍼 이상', '캐시 비정상'],
        '파일': ['11_SMS_메모리_5min.csv'] * 4
    }
    st.dataframe(pd.DataFrame(mem_data), width='stretch')

    st.subheader("3️⃣ 파일시스템 이상 탐지")
    fs_data = {
        '열': ['USED_PCT', 'AVAIL_SIZE', 'INODE_USED_PCT'],
        '설명': ['디스크 사용률', '가용 공간', 'Inode 사용률'],
        '적합 알고리즘': ['임계값(80/90%), 추세 분석', 'IQR (급감 탐지)', 'Z-Score'],
        '이상 시나리오': ['디스크 풀', '공간 급감', 'Inode 고갈'],
        '파일': ['12_SMS_파일시스템_5min.csv'] * 3
    }
    st.dataframe(pd.DataFrame(fs_data), width='stretch')

    st.markdown("---")

    # ===== 2. NMS 네트워크 모니터링 =====
    st.header("🌐 NMS 네트워크 모니터링")

    st.subheader("4️⃣ 트래픽 이상 탐지")
    traffic_data = {
        '열': ['AVG_INBPS', 'AVG_OUTBPS', 'AVG_INPPS', 'AVG_OUTPPS', 'INBPS_RATE', 'OUTBPS_RATE'],
        '설명': ['인바운드 트래픽(bps)', '아웃바운드 트래픽(bps)', '인바운드 패킷(pps)', '아웃바운드 패킷(pps)', '인바운드 대역폭 사용률', '아웃바운드 대역폭 사용률'],
        '적합 알고리즘': ['Isolation Forest, Z-Score', 'Isolation Forest, Z-Score', 'Z-Score', 'Z-Score', '임계값(80/90%)', '임계값(80/90%)'],
        '이상 시나리오': ['DDoS, 대용량 전송', '데이터 유출, 백업', '패킷 폭주', '패킷 폭주', '대역폭 포화', '대역폭 포화'],
        '파일': ['04_NMS_IF성능_5min.csv'] * 6
    }
    st.dataframe(pd.DataFrame(traffic_data), width='stretch')

    st.subheader("5️⃣ 네트워크 오류 탐지")
    error_data = {
        '열': ['AVG_INERR', 'AVG_OUTERR', 'AVG_CRC', 'AVG_COLLISION', 'AVG_INDROP', 'AVG_OUTDROP', 'AVG_INDISCARD', 'AVG_OUTDISCARD'],
        '설명': ['인바운드 에러', '아웃바운드 에러', 'CRC 에러', '충돌', '인바운드 드롭', '아웃바운드 드롭', '인바운드 폐기', '아웃바운드 폐기'],
        '적합 알고리즘': ['Z-Score (0이 정상)', 'Z-Score (0이 정상)', 'Z-Score (케이블/NIC 이슈)', 'Z-Score (허브환경)', 'Z-Score (버퍼 부족)', 'Z-Score (버퍼 부족)', 'Z-Score (QoS)', 'Z-Score (QoS)'],
        '이상 시나리오': ['패킷 손상', '패킷 손상', '물리적 문제', '네트워크 혼잡', '수신 버퍼 오버플로', '송신 버퍼 오버플로', '정책 폐기', '정책 폐기'],
        '파일': ['04_NMS_IF성능_5min.csv'] * 8
    }
    st.dataframe(pd.DataFrame(error_data), width='stretch')

    st.markdown("---")

    # ===== 3. 다변량 결합 분석 =====
    st.header("🔗 다변량 결합 분석 (Isolation Forest)")

    st.markdown("""
    **Isolation Forest**는 여러 변수를 동시에 고려하여 단변량으로는 발견하기 어려운 복합 이상을 탐지합니다.
    """)

    combo_data = {
        '결합 변수': [
            'CPU_USAGE_AVG + PHYSICAL_USED_PCT',
            'CPU_USAGE_AVG + IOWAIT_PCT_AVG',
            'AVG_INBPS + AVG_OUTBPS',
            'AVG_INBPS + AVG_INERR',
            'USED_PCT + CPU_USAGE_AVG',
            'CPU_USAGE_AVG + RUN_QUEUE_AVG',
            'AVG_INBPS + AVG_INPPS'
        ],
        '결합 파일': [
            '10_SMS_CPU + 11_SMS_메모리',
            '10_SMS_CPU (내부)',
            '04_NMS_IF성능 (내부)',
            '04_NMS_IF성능 (내부)',
            '12_SMS_파일시스템 + 10_SMS_CPU',
            '10_SMS_CPU (내부)',
            '04_NMS_IF성능 (내부)'
        ],
        '결합 키': [
            'YYYYMMDD, TIME_ID, MNG_NO',
            '동일 레코드',
            '동일 레코드',
            '동일 레코드',
            'YYYYMMDD, TIME_ID, MNG_NO',
            '동일 레코드',
            '동일 레코드'
        ],
        '탐지 목적': [
            'CPU↑ + 메모리↑ = 과부하',
            'CPU↑ + I/O↑ = 디스크 병목',
            'IN↑ + OUT↑ = 정상 통신 vs 한쪽만↑ = 이상',
            '트래픽↑ + 에러↑ = 네트워크 장애',
            '디스크↑ + CPU↑ = 로그/스왑 이슈',
            'CPU↑ + 큐↑ = 심각한 과부하',
            'BPS↑ + PPS↓ = 대용량 패킷 (정상) vs 둘다↑ = 공격'
        ],
        '알고리즘': ['Isolation Forest'] * 7
    }
    st.dataframe(pd.DataFrame(combo_data), width='stretch')

    st.markdown("---")

    # ===== 4. 시계열 이상 탐지 =====
    st.header("📈 시계열 기반 이상 탐지")

    ts_data = {
        '방법': ['이동평균 편차', '계절성 분해', '변화율 탐지', 'ARIMA 잔차'],
        '설명': ['최근 N개 평균 대비 현재값 비교', '시간/요일/월 패턴 분해 후 잔차 분석', '전 시점 대비 급격한 변화', '예측값 대비 실제값 편차'],
        '적용 열': ['CPU, 메모리, 트래픽 등 모든 수치형', '주기성 있는 데이터', '모든 수치형', 'CPU, 트래픽 등'],
        '장점': ['간단, 실시간 적용 가능', '정상 패턴 학습 가능', '급격한 변화 즉시 탐지', '트렌드/계절성 고려'],
        '구현 난이도': ['쉬움', '중간', '쉬움', '어려움']
    }
    st.dataframe(pd.DataFrame(ts_data), width='stretch')

    st.markdown("---")

    # ===== 5. 알고리즘별 적합 데이터 =====
    st.header("🧮 알고리즘별 적합 데이터 요약")

    algo_summary = {
        '알고리즘': ['Z-Score', 'IQR', '임계값 기반', 'Isolation Forest', '변화율 탐지'],
        '적합 데이터': [
            '정규분포에 가까운 데이터 (CPU, 메모리 사용률)',
            '비대칭 분포, 이상치에 강건해야 할 때',
            '운영 기준이 명확한 경우 (사용률 80/90%)',
            '다변량 복합 패턴, 비선형 이상',
            '급격한 변화가 문제인 경우'
        ],
        '장점': [
            '해석 용이, 계산 빠름',
            '분포 가정 없음, 강건함',
            '직관적, 운영 기준과 일치',
            '복합 이상 탐지, 고차원 데이터',
            '실시간 탐지 가능'
        ],
        '단점': [
            '정규분포 가정 필요',
            '극단값 정의 고정적',
            '점진적 이상 탐지 어려움',
            '블랙박스, 해석 어려움',
            '정상적인 급변도 탐지'
        ],
        '추천 열': [
            'CPU_USAGE_AVG, PHYSICAL_USED_PCT, AVG_INBPS',
            'AVG_CRC, AVG_COLLISION, RUN_QUEUE',
            '사용률(%), 대역폭 사용률',
            'CPU+메모리, IN+OUT 트래픽',
            '모든 연속형 지표'
        ]
    }
    st.dataframe(pd.DataFrame(algo_summary), width='stretch')

    st.markdown("---")

    # ===== 6. 실제 구현 예시 =====
    st.header("💻 구현 예시 코드")

    with st.expander("Z-Score 이상 탐지"):
        st.code('''
from scipy import stats
import numpy as np

def detect_zscore(series, threshold=3.0):
    """Z-Score 기반 이상 탐지"""
    z_scores = np.abs(stats.zscore(series.fillna(series.median())))
    return z_scores > threshold

# 사용 예시
df['is_anomaly'] = detect_zscore(df['CPU_USAGE_AVG'], threshold=3.0)
        ''', language='python')

    with st.expander("Isolation Forest 다변량 이상 탐지"):
        st.code('''
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_isolation_forest(df, feature_cols, contamination=0.05):
    """Isolation Forest 기반 다변량 이상 탐지"""
    features = df[feature_cols].fillna(df[feature_cols].median())
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(features_scaled)

    return predictions == -1  # -1이 이상치

# 사용 예시 (CPU + 메모리 결합)
df_merged = df_cpu.merge(df_memory, on=['YYYYMMDD', 'TIME_ID', 'MNG_NO'])
df_merged['is_anomaly'] = detect_isolation_forest(
    df_merged,
    ['CPU_USAGE_AVG', 'PHYSICAL_USED_PCT']
)
        ''', language='python')

    with st.expander("변화율 기반 이상 탐지"):
        st.code('''
def detect_change_rate(series, threshold=50):
    """변화율 기반 이상 탐지 (전 시점 대비 %)"""
    pct_change = series.pct_change().abs() * 100
    return pct_change > threshold

# 사용 예시 (50% 이상 급변 탐지)
df['is_spike'] = detect_change_rate(df['AVG_INBPS'], threshold=50)
        ''', language='python')

    st.markdown("---")

    # ===== 7. 권장 이상 탐지 파이프라인 =====
    st.header("🔄 권장 이상 탐지 파이프라인")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    이상 탐지 파이프라인                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1️⃣ 1차 필터: 임계값 기반                                       │
    │     └─ CPU ≥ 80%, 메모리 ≥ 80%, 디스크 ≥ 80%                   │
    │                    ↓                                            │
    │  2️⃣ 2차 분석: 통계 기반 (Z-Score)                              │
    │     └─ 3σ 초과 데이터 추출                                      │
    │                    ↓                                            │
    │  3️⃣ 3차 분석: 다변량 (Isolation Forest)                        │
    │     └─ CPU + 메모리 + I/O 복합 분석                             │
    │                    ↓                                            │
    │  4️⃣ 4차 분석: 시계열 패턴                                       │
    │     └─ 급격한 변화, 비정상 패턴 탐지                            │
    │                    ↓                                            │
    │  5️⃣ 결과 통합 및 알림                                          │
    │     └─ 심각도 분류, 근본 원인 분석, 대응 권고                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)


def page_raw_eda():
    """각 CSV 파일별 기초 EDA 페이지"""
    import glob as glob_module
    import re

    st.title("📁 원본 데이터 탐색")
    st.markdown("**각 CSV 파일별 기초 EDA**")

    # 컬럼 설명 로드
    column_descriptions = parse_column_descriptions()

    # 폴더별 CSV 파일 수집
    output_files = sorted(glob_module.glob('./output/*.csv'))
    dbtable_files = sorted(glob_module.glob('./dbtable/*.csv'))

    # 파일명만 추출하여 딕셔너리 생성
    csv_files = {}

    for f in output_files:
        name = f.replace('\\', '/').split('/')[-1].replace('.csv', '')
        csv_files[f"[output] {name}"] = f

    for f in dbtable_files:
        name = f.replace('\\', '/').split('/')[-1].replace('.csv', '')
        csv_files[f"[dbtable] {name}"] = f

    if not csv_files:
        st.error("CSV 파일을 찾을 수 없습니다.")
        return

    # 폴더 필터
    col1, col2 = st.columns([1, 3])

    with col1:
        folder_filter = st.radio("폴더", ["전체", "output", "dbtable"], horizontal=True)

    # 필터 적용
    if folder_filter == "output":
        filtered_files = {k: v for k, v in csv_files.items() if k.startswith("[output]")}
    elif folder_filter == "dbtable":
        filtered_files = {k: v for k, v in csv_files.items() if k.startswith("[dbtable]")}
    else:
        filtered_files = csv_files

    with col2:
        selected_file = st.selectbox(
            f"📂 CSV 파일 선택 ({len(filtered_files)}개)",
            list(filtered_files.keys())
        )

    if not selected_file:
        return

    # 파일 로드
    file_path = filtered_files[selected_file]

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='cp949')
        except Exception as e:
            st.error(f"파일 로드 실패: {e}")
            return
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return

    st.caption(f"📍 경로: `{file_path}`")

    st.markdown("---")

    # ===== 1. 기본 정보 =====
    st.subheader("1️⃣ 기본 정보")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("행 수", f"{len(df):,}")
    with col2:
        st.metric("열 수", f"{len(df.columns)}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("메모리", f"{memory_mb:.2f} MB")
    with col4:
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("결측률", f"{null_pct:.2f}%")

    # ===== 2. 컬럼 정보 =====
    st.subheader("2️⃣ 컬럼 정보")

    # 테이블명 추출 (파일명에서)
    file_basename = file_path.replace('\\', '/').split('/')[-1].replace('.csv', '')

    # 컬럼 설명 찾기
    table_col_desc = {}

    # output 파일: "01_NMS_장비마스터" 형식
    if file_basename in column_descriptions:
        table_col_desc = column_descriptions[file_basename]
    else:
        # dbtable 파일: 직접 테이블명으로 찾기
        for table_name, cols in column_descriptions.items():
            if table_name == file_basename or file_basename.startswith(table_name):
                table_col_desc = cols
                break

    col_info = pd.DataFrame({
        '컬럼명': df.columns,
        '데이터타입': df.dtypes.astype(str).values,
        '결측수': df.isnull().sum().values,
        '결측률(%)': (df.isnull().sum().values / len(df) * 100).round(2),
        '고유값수': df.nunique().values
    })

    # 컬럼 설명 추가
    descriptions = []
    for col in df.columns:
        desc = table_col_desc.get(col, '')
        descriptions.append(desc)
    col_info['설명'] = descriptions

    # 샘플값 추가
    sample_values = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            sample = str(non_null.iloc[0])[:30]
            sample_values.append(sample + ('...' if len(str(non_null.iloc[0])) > 30 else ''))
        else:
            sample_values.append('(없음)')
    col_info['샘플값'] = sample_values

    # 컬럼 순서 재정렬
    col_info = col_info[['컬럼명', '설명', '데이터타입', '결측수', '결측률(%)', '고유값수', '샘플값']]

    st.dataframe(col_info, width='stretch', height=300)

    # ===== 3. 결측치 시각화 =====
    st.subheader("3️⃣ 결측치 현황")

    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0].sort_values(ascending=False)

    if len(null_cols) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                x=null_cols.index[:20],
                y=null_cols.values[:20],
                title=f"결측치 컬럼 (상위 {min(20, len(null_cols))}개)",
                labels={'x': '컬럼', 'y': '결측 수'},
                color=null_cols.values[:20],
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("#### 결측 컬럼 요약")
            st.markdown(f"- 결측 있는 컬럼: **{len(null_cols)}개**")
            st.markdown(f"- 총 결측 셀: **{null_cols.sum():,}개**")
            if len(null_cols) > 0:
                st.markdown(f"- 최다 결측: **{null_cols.index[0]}** ({null_cols.iloc[0]:,}건)")
    else:
        st.success("✅ 결측치가 없습니다!")

    # ===== 4. 수치형 컬럼 통계 =====
    st.subheader("4️⃣ 수치형 컬럼 기본 통계")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe().T
        stats_df.columns = ['개수', '평균', '표준편차', '최소', '25%', '50%', '75%', '최대']
        st.dataframe(stats_df.round(2), width='stretch', height=250)

        # 분포 시각화
        st.markdown("#### 분포 시각화")

        # 컬럼 선택
        viz_cols = st.multiselect(
            "시각화할 컬럼 선택 (최대 4개)",
            numeric_cols,
            default=numeric_cols[:min(2, len(numeric_cols))]
        )

        if viz_cols:
            cols = st.columns(min(len(viz_cols), 4))
            for i, col_name in enumerate(viz_cols[:4]):
                with cols[i % 4]:
                    fig = px.histogram(
                        df, x=col_name, nbins=30,
                        title=f"{col_name}",
                        color_discrete_sequence=['#3498db']
                    )
                    fig.update_layout(height=250, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
    else:
        st.info("수치형 컬럼이 없습니다.")

    # ===== 5. 범주형 컬럼 분석 =====
    st.subheader("5️⃣ 범주형 컬럼 분석")

    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    if len(cat_cols) > 0:
        selected_cat = st.selectbox("범주형 컬럼 선택", cat_cols)

        col1, col2 = st.columns(2)

        with col1:
            value_counts = df[selected_cat].value_counts().head(15)
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index.astype(str),
                orientation='h',
                title=f"{selected_cat} 값 분포 (상위 15개)",
                labels={'x': '건수', 'y': ''},
                color_discrete_sequence=['#2980b9']  # 진한 파란색
            )
            fig.update_traces(
                text=value_counts.values,
                textposition='outside'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown(f"#### {selected_cat} 요약")
            st.markdown(f"- 고유값 수: **{df[selected_cat].nunique():,}개**")
            st.markdown(f"- 결측 수: **{df[selected_cat].isnull().sum():,}건**")
            st.markdown(f"- 최빈값: **{df[selected_cat].mode().iloc[0] if len(df[selected_cat].mode()) > 0 else 'N/A'}**")

            st.markdown("#### 상위 5개 값")
            top5 = df[selected_cat].value_counts().head(5)
            for val, cnt in top5.items():
                pct = cnt / len(df) * 100
                st.markdown(f"- {val}: {cnt:,}건 ({pct:.1f}%)")
    else:
        st.info("범주형 컬럼이 없습니다.")

    # ===== 6. 샘플 데이터 =====
    st.subheader("6️⃣ 샘플 데이터")

    sample_size = st.slider("샘플 크기", 5, 50, 10)

    tab1, tab2, tab3 = st.tabs(["처음 N행", "마지막 N행", "랜덤 샘플"])

    with tab1:
        st.dataframe(df.head(sample_size), width='stretch')
    with tab2:
        st.dataframe(df.tail(sample_size), width='stretch')
    with tab3:
        st.dataframe(df.sample(min(sample_size, len(df))), width='stretch')

    # ===== 7. 중복 데이터 =====
    st.subheader("7️⃣ 중복 데이터 분석")

    col1, col2 = st.columns(2)

    with col1:
        dup_count = df.duplicated().sum()
        st.metric("완전 중복 행", f"{dup_count:,}건",
                 f"{dup_count/len(df)*100:.2f}%" if len(df) > 0 else "0%")

    with col2:
        # 키 컬럼 중복 확인
        if 'MNG_NO' in df.columns:
            if 'YMDHMS' in df.columns:
                key_dup = df.duplicated(subset=['MNG_NO', 'YMDHMS']).sum()
                st.metric("키 중복 (MNG_NO + YMDHMS)", f"{key_dup:,}건")
            elif 'YYYYMMDD' in df.columns:
                key_dup = df.duplicated(subset=['MNG_NO', 'YYYYMMDD']).sum()
                st.metric("키 중복 (MNG_NO + YYYYMMDD)", f"{key_dup:,}건")
            else:
                mng_dup = df.duplicated(subset=['MNG_NO']).sum()
                st.metric("MNG_NO 중복", f"{mng_dup:,}건")


# ============== EDA 페이지 ==============
def page_eda(data):
    """탐색적 데이터 분석 페이지"""
    st.title("🔍 NETIS 데이터 탐색")
    st.markdown("**데이터 현황 및 탐색적 분석 (EDA)**")

    # ============== 데이터 개요 카드 ==============
    st.markdown("---")

    # 주요 지표 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nms_count = len(data.get('nms_master', []))
        st.metric("🌐 NMS 장비", f"{nms_count:,}대")

    with col2:
        sms_count = len(data.get('sms_master', []))
        st.metric("🖥️ SMS 서버", f"{sms_count:,}대")

    with col3:
        if_perf_count = len(data.get('nms_if_perf', []))
        st.metric("📶 NMS 성능 데이터", f"{if_perf_count:,}건")

    with col4:
        cpu_count = len(data.get('sms_cpu', []))
        st.metric("💻 SMS CPU 데이터", f"{cpu_count:,}건")

    # 데이터 기간 표시
    if 'nms_if_perf' in data and 'DATETIME' in data['nms_if_perf'].columns:
        df = data['nms_if_perf']
        date_min = df['DATETIME'].min()
        date_max = df['DATETIME'].max()
        st.markdown(f"""
        <div class="alert-normal">
            <strong>📅 데이터 수집 기간:</strong> {date_min} ~ {date_max}
        </div>
        """, unsafe_allow_html=True)

    # ============== 탭 기반 분석 ==============
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 장비 현황", "📈 리소스 분석", "🔗 상관관계",
        "📉 시계열 패턴", "🔬 이상치 분석"
    ])

    # -------------- 탭1: 장비 현황 --------------
    with tab1:
        st.header("📊 장비 및 서버 현황")

        # NMS 장비 분석
        st.subheader("🌐 NMS 네트워크 장비")

        if 'nms_master' in data:
            df_nms = data['nms_master']

            # 메트릭 카드
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'DEV_KIND1' in df_nms.columns:
                    kind_count = df_nms['DEV_KIND1'].nunique()
                    st.metric("장비 종류", f"{kind_count}종")
            with col2:
                if 'VENDOR' in df_nms.columns:
                    vendor_count = df_nms['VENDOR'].nunique()
                    st.metric("제조사", f"{vendor_count}개사")
            with col3:
                if 'MODEL' in df_nms.columns:
                    model_count = df_nms['MODEL'].nunique()
                    st.metric("모델", f"{model_count}종")

            col1, col2 = st.columns(2)

            with col1:
                if 'DEV_KIND1' in df_nms.columns:
                    kind_counts = df_nms['DEV_KIND1'].value_counts()
                    fig = px.pie(
                        values=kind_counts.values,
                        names=kind_counts.index,
                        title="장비 종류별 분포",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, width='stretch')

            with col2:
                if 'VENDOR' in df_nms.columns:
                    vendor_counts = df_nms['VENDOR'].value_counts().head(8)
                    fig = px.bar(
                        x=vendor_counts.values,
                        y=vendor_counts.index,
                        orientation='h',
                        title="제조사별 장비 수 (TOP 8)",
                        labels={'x': '장비 수', 'y': ''},
                        color=vendor_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # SMS 서버 분석
        st.subheader("🖥️ SMS 서버")

        if 'sms_master' in data:
            df_sms = data['sms_master']

            col1, col2 = st.columns(2)

            with col1:
                if 'DEV_KIND2' in df_sms.columns:
                    kind_counts = df_sms['DEV_KIND2'].value_counts()
                    fig = px.pie(
                        values=kind_counts.values,
                        names=kind_counts.index,
                        title="서버 종류별 분포",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, width='stretch')

            with col2:
                if 'VENDOR' in df_sms.columns:
                    vendor_counts = df_sms['VENDOR'].value_counts()
                    fig = px.bar(
                        x=vendor_counts.index,
                        y=vendor_counts.values,
                        title="서버 제조사별 분포",
                        labels={'x': '', 'y': '서버 수'},
                        color_discrete_sequence=['#27ae60']  # 단일 진한 초록색
                    )
                    fig.update_traces(
                        text=vendor_counts.values,
                        textposition='outside'
                    )
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, width='stretch')

    # -------------- 탭2: 리소스 분석 --------------
    with tab2:
        st.header("📈 리소스 사용률 분석")

        # CPU 분석
        st.subheader("💻 CPU 사용률")

        if 'sms_cpu' in data and 'CPU_USAGE_AVG' in data['sms_cpu'].columns:
            df_cpu = data['sms_cpu']

            # 요약 지표
            col1, col2, col3, col4 = st.columns(4)
            cpu_stats = df_cpu['CPU_USAGE_AVG'].describe()

            with col1:
                st.metric("평균", f"{cpu_stats['mean']:.1f}%")
            with col2:
                st.metric("최대", f"{cpu_stats['max']:.1f}%")
            with col3:
                warning_pct = (df_cpu['CPU_USAGE_AVG'] >= 80).mean() * 100
                st.metric("경고 수준 비율", f"{warning_pct:.1f}%",
                         delta=f"80% 이상", delta_color="inverse" if warning_pct > 5 else "normal")
            with col4:
                critical_pct = (df_cpu['CPU_USAGE_AVG'] >= 90).mean() * 100
                st.metric("위험 수준 비율", f"{critical_pct:.1f}%",
                         delta=f"90% 이상", delta_color="inverse" if critical_pct > 1 else "normal")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    df_cpu, x='CPU_USAGE_AVG', nbins=50,
                    title="CPU 사용률 분포",
                    labels={'CPU_USAGE_AVG': 'CPU 사용률 (%)'},
                    color_discrete_sequence=['#3498db']
                )
                fig.add_vline(x=80, line_dash="dash", line_color="#f39c12", line_width=2,
                             annotation_text="경고 80%", annotation_position="top")
                fig.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=2,
                             annotation_text="위험 90%", annotation_position="top")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            with col2:
                # 서버별 통계 계산
                server_stats = df_cpu.groupby('MNG_NO')['CPU_USAGE_AVG'].agg(['mean', 'std', 'max']).reset_index()
                server_stats.columns = ['서버', '평균', '표준편차', '최대']

                # 위험도 분류
                server_stats['상태'] = server_stats['평균'].apply(
                    lambda x: '위험' if x >= 90 else '경고' if x >= 80 else '정상'
                )

                fig = px.scatter(
                    server_stats,
                    x='평균',
                    y='최대',
                    size='표준편차',
                    color='상태',
                    color_discrete_map={'정상': '#3498db', '경고': '#f39c12', '위험': '#e74c3c'},
                    hover_name='서버',
                    hover_data={'평균': ':.1f', '최대': ':.1f', '표준편차': ':.1f'},
                    title="서버별 CPU: 평균 vs 최대 (크기=변동성)"
                )

                # 경고/위험 영역
                fig.add_vline(x=80, line_dash="dash", line_color="#f39c12", line_width=1)
                fig.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=1)
                fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", line_width=1)
                fig.add_hline(y=90, line_dash="dash", line_color="#e74c3c", line_width=1)

                fig.update_layout(
                    height=350,
                    xaxis_title="평균 CPU (%)",
                    yaxis_title="최대 CPU (%)",
                    xaxis=dict(range=[0, 105]),
                    yaxis=dict(range=[0, 105])
                )
                st.plotly_chart(fig, width='stretch')

            # 시간대별 추이
            if 'DATETIME' in df_cpu.columns:
                df_hourly = df_cpu.copy()
                df_hourly['HOUR'] = df_hourly['DATETIME'].dt.hour
                hourly_stats = df_hourly.groupby('HOUR')['CPU_USAGE_AVG'].agg(['mean', 'std']).reset_index()
                hourly_stats['upper'] = hourly_stats['mean'] + hourly_stats['std']
                hourly_stats['lower'] = (hourly_stats['mean'] - hourly_stats['std']).clip(lower=0)

                fig = go.Figure()

                # 평균 + 표준편차 (상단 경계)
                fig.add_trace(go.Scatter(
                    x=hourly_stats['HOUR'], y=hourly_stats['upper'],
                    mode='lines', name='+1 표준편차',
                    line=dict(color='rgba(52, 152, 219, 0.3)', width=0),
                    showlegend=False
                ))

                # 평균 - 표준편차 (하단 경계) + 영역 채우기
                fig.add_trace(go.Scatter(
                    x=hourly_stats['HOUR'], y=hourly_stats['lower'],
                    mode='lines', name='±1 표준편차',
                    line=dict(color='rgba(52, 152, 219, 0.3)', width=0),
                    fill='tonexty', fillcolor='rgba(52, 152, 219, 0.25)'
                ))

                # 평균선
                fig.add_trace(go.Scatter(
                    x=hourly_stats['HOUR'], y=hourly_stats['mean'],
                    mode='lines+markers', name='평균',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=6)
                ))

                fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", line_width=1,
                             annotation_text="경고")
                fig.update_layout(
                    title="시간대별 CPU 사용률 추이 (평균 ± 표준편차)",
                    xaxis_title="시간 (시)",
                    yaxis_title="CPU 사용률 (%)",
                    height=350,
                    xaxis=dict(tickmode='linear', dtick=2)
                )
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # 메모리 분석
        st.subheader("🧠 메모리 사용률")

        if 'sms_memory' in data and 'PHYSICAL_USED_PCT' in data['sms_memory'].columns:
            df_mem = data['sms_memory']

            # 요약 지표
            col1, col2, col3, col4 = st.columns(4)
            mem_stats = df_mem['PHYSICAL_USED_PCT'].describe()

            with col1:
                st.metric("평균", f"{mem_stats['mean']:.1f}%")
            with col2:
                st.metric("최대", f"{mem_stats['max']:.1f}%")
            with col3:
                warning_pct = (df_mem['PHYSICAL_USED_PCT'] >= 80).mean() * 100
                st.metric("경고 수준 비율", f"{warning_pct:.1f}%")
            with col4:
                critical_pct = (df_mem['PHYSICAL_USED_PCT'] >= 90).mean() * 100
                st.metric("위험 수준 비율", f"{critical_pct:.1f}%")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    df_mem, x='PHYSICAL_USED_PCT', nbins=50,
                    title="메모리 사용률 분포",
                    labels={'PHYSICAL_USED_PCT': '메모리 사용률 (%)'},
                    color_discrete_sequence=['#9b59b6']
                )
                fig.add_vline(x=80, line_dash="dash", line_color="#f39c12", line_width=2)
                fig.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=2)
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            with col2:
                # 서버별 통계 계산
                server_stats = df_mem.groupby('MNG_NO')['PHYSICAL_USED_PCT'].agg(['mean', 'std', 'max']).reset_index()
                server_stats.columns = ['서버', '평균', '표준편차', '최대']

                # 위험도 분류
                server_stats['상태'] = server_stats['평균'].apply(
                    lambda x: '위험' if x >= 90 else '경고' if x >= 80 else '정상'
                )

                fig = px.scatter(
                    server_stats,
                    x='평균',
                    y='최대',
                    size='표준편차',
                    color='상태',
                    color_discrete_map={'정상': '#9b59b6', '경고': '#f39c12', '위험': '#e74c3c'},
                    hover_name='서버',
                    hover_data={'평균': ':.1f', '최대': ':.1f', '표준편차': ':.1f'},
                    title="서버별 메모리: 평균 vs 최대 (크기=변동성)"
                )

                # 경고/위험 영역
                fig.add_vline(x=80, line_dash="dash", line_color="#f39c12", line_width=1)
                fig.add_vline(x=90, line_dash="dash", line_color="#e74c3c", line_width=1)
                fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", line_width=1)
                fig.add_hline(y=90, line_dash="dash", line_color="#e74c3c", line_width=1)

                fig.update_layout(
                    height=350,
                    xaxis_title="평균 메모리 (%)",
                    yaxis_title="최대 메모리 (%)",
                    xaxis=dict(range=[0, 105]),
                    yaxis=dict(range=[0, 105])
                )
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # 네트워크 트래픽 분석
        st.subheader("📶 네트워크 트래픽")

        if 'nms_if_perf' in data:
            df_if = data['nms_if_perf']

            # 샘플링
            if len(df_if) > 50000:
                df_if_sample = df_if.sample(n=50000, random_state=42)
                st.caption("⚠️ 대용량 데이터: 50,000건 샘플링 분석")
            else:
                df_if_sample = df_if

            # 트래픽 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'AVG_INBPS' in df_if.columns:
                    avg_in = df_if['AVG_INBPS'].mean()
                    st.metric("평균 IN 트래픽", f"{avg_in/1e6:.2f} Mbps")
            with col2:
                if 'AVG_OUTBPS' in df_if.columns:
                    avg_out = df_if['AVG_OUTBPS'].mean()
                    st.metric("평균 OUT 트래픽", f"{avg_out/1e6:.2f} Mbps")
            with col3:
                if 'AVG_INERR' in df_if.columns:
                    total_err = df_if['AVG_INERR'].sum()
                    st.metric("총 IN 에러", f"{total_err:,.0f}")
            with col4:
                if 'AVG_OUTERR' in df_if.columns:
                    total_err = df_if['AVG_OUTERR'].sum()
                    st.metric("총 OUT 에러", f"{total_err:,.0f}")

            col1, col2 = st.columns(2)

            with col1:
                if 'AVG_INBPS' in df_if_sample.columns:
                    df_if_sample['LOG_INBPS'] = np.log10(df_if_sample['AVG_INBPS'].replace(0, 1))
                    fig = px.histogram(
                        df_if_sample, x='LOG_INBPS', nbins=50,
                        title="인바운드 트래픽 분포 (로그 스케일)",
                        labels={'LOG_INBPS': 'log₁₀(IN BPS)'},
                        color_discrete_sequence=['#1abc9c']
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, width='stretch')

            with col2:
                if 'AVG_OUTBPS' in df_if_sample.columns:
                    df_if_sample['LOG_OUTBPS'] = np.log10(df_if_sample['AVG_OUTBPS'].replace(0, 1))
                    fig = px.histogram(
                        df_if_sample, x='LOG_OUTBPS', nbins=50,
                        title="아웃바운드 트래픽 분포 (로그 스케일)",
                        labels={'LOG_OUTBPS': 'log₁₀(OUT BPS)'},
                        color_discrete_sequence=['#e67e22']
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, width='stretch')

    # -------------- 탭3: 상관관계 --------------
    with tab3:
        st.header("🔗 상관관계 분석")

        # CPU vs 메모리
        st.subheader("💻 CPU vs 🧠 메모리")

        if 'sms_cpu' in data and 'sms_memory' in data:
            df_cpu = data['sms_cpu']
            df_mem = data['sms_memory']

            if 'CPU_USAGE_AVG' in df_cpu.columns and 'PHYSICAL_USED_PCT' in df_mem.columns:
                df_combined = df_cpu.merge(
                    df_mem[['YYYYMMDD', 'TIME_ID', 'MNG_NO', 'PHYSICAL_USED_PCT']],
                    on=['YYYYMMDD', 'TIME_ID', 'MNG_NO'],
                    how='inner'
                )

                if len(df_combined) > 0:
                    # 상관계수
                    corr = df_combined['CPU_USAGE_AVG'].corr(df_combined['PHYSICAL_USED_PCT'])

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 30px; border-radius: 15px; text-align: center; color: white;">
                            <h2 style="margin: 0; font-size: 2.5rem;">{corr:.3f}</h2>
                            <p style="margin: 10px 0 0 0;">Pearson 상관계수</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("")

                        # 상관관계 해석
                        if abs(corr) >= 0.7:
                            interp = "강한 상관관계"
                            color = "#e74c3c"
                        elif abs(corr) >= 0.4:
                            interp = "중간 상관관계"
                            color = "#f39c12"
                        else:
                            interp = "약한 상관관계"
                            color = "#27ae60"

                        st.markdown(f"""
                        <div style="background-color: {color}20; border-left: 4px solid {color};
                                    padding: 15px; border-radius: 5px;">
                            <strong>{interp}</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # 샘플링하여 산점도
                        sample_size = min(5000, len(df_combined))
                        df_sample = df_combined.sample(n=sample_size, random_state=42)

                        fig = px.scatter(
                            df_sample,
                            x='CPU_USAGE_AVG',
                            y='PHYSICAL_USED_PCT',
                            color='MNG_NO',
                            opacity=0.6,
                            title="CPU vs 메모리 사용률 (서버별)",
                            labels={
                                'CPU_USAGE_AVG': 'CPU 사용률 (%)',
                                'PHYSICAL_USED_PCT': '메모리 사용률 (%)',
                                'MNG_NO': '서버'
                            },
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )

                        # 경고 구역 표시
                        fig.add_vrect(x0=80, x1=100, fillcolor="#f39c12", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=80, y1=100, fillcolor="#f39c12", opacity=0.1, line_width=0)
                        fig.add_vline(x=80, line_dash="dash", line_color="#f39c12", line_width=1)
                        fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", line_width=1)

                        fig.update_layout(height=450)
                        st.plotly_chart(fig, width='stretch')

        st.markdown("---")

        # IN vs OUT 트래픽
        st.subheader("📥 IN vs 📤 OUT 트래픽")

        if 'nms_if_perf' in data:
            df_if = data['nms_if_perf']

            if 'AVG_INBPS' in df_if.columns and 'AVG_OUTBPS' in df_if.columns:
                # 상관계수
                corr_traffic = df_if['AVG_INBPS'].corr(df_if['AVG_OUTBPS'])

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                                padding: 30px; border-radius: 15px; text-align: center; color: white;">
                        <h2 style="margin: 0; font-size: 2.5rem;">{corr_traffic:.3f}</h2>
                        <p style="margin: 10px 0 0 0;">Pearson 상관계수</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    sample_size = min(5000, len(df_if))
                    df_sample = df_if.sample(n=sample_size, random_state=42)

                    fig = px.scatter(
                        df_sample,
                        x='AVG_INBPS',
                        y='AVG_OUTBPS',
                        opacity=0.4,
                        title="IN vs OUT 트래픽 상관관계",
                        labels={'AVG_INBPS': 'IN BPS', 'AVG_OUTBPS': 'OUT BPS'},
                        color_discrete_sequence=['#1abc9c']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')

    # -------------- 탭4: 시계열 패턴 --------------
    with tab4:
        st.header("📉 시계열 패턴 분석")
        st.markdown("*정상 패턴을 파악하여 이상 탐지 기준 수립에 활용*")

        if 'sms_cpu' in data and 'DATETIME' in data['sms_cpu'].columns:
            df_cpu = data['sms_cpu'].copy()

            # 시간 특성 추출
            df_cpu['HOUR'] = df_cpu['DATETIME'].dt.hour
            df_cpu['DAYOFWEEK'] = df_cpu['DATETIME'].dt.dayofweek
            df_cpu['DAY_NAME'] = df_cpu['DATETIME'].dt.day_name()
            df_cpu['DATE'] = df_cpu['DATETIME'].dt.date

            # 요일별 패턴
            st.subheader("📅 요일별 CPU 사용 패턴")

            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_korean = {'Monday': '월', 'Tuesday': '화', 'Wednesday': '수',
                         'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'}

            daily_stats = df_cpu.groupby('DAY_NAME')['CPU_USAGE_AVG'].agg(['mean', 'std', 'max']).reset_index()
            daily_stats['DAY_NAME'] = pd.Categorical(daily_stats['DAY_NAME'], categories=day_order, ordered=True)
            daily_stats = daily_stats.sort_values('DAY_NAME')
            daily_stats['요일'] = daily_stats['DAY_NAME'].map(day_korean)

            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=daily_stats['요일'],
                    y=daily_stats['mean'],
                    name='평균',
                    marker_color='#3498db',
                    error_y=dict(type='data', array=daily_stats['std'], visible=True)
                ))
                fig.update_layout(
                    title="요일별 평균 CPU 사용률 (±표준편차)",
                    xaxis_title="요일",
                    yaxis_title="CPU 사용률 (%)",
                    height=350
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # 요일 × 시간 히트맵
                heatmap_data = df_cpu.groupby(['DAYOFWEEK', 'HOUR'])['CPU_USAGE_AVG'].mean().reset_index()
                heatmap_pivot = heatmap_data.pivot(index='DAYOFWEEK', columns='HOUR', values='CPU_USAGE_AVG')

                # 모든 시간(0~23)과 요일(0~6)을 포함하도록 reindex
                heatmap_pivot = heatmap_pivot.reindex(
                    index=range(7),
                    columns=range(24),
                    fill_value=np.nan
                )

                day_labels = ['월', '화', '수', '목', '금', '토', '일']
                hour_labels = [f"{h}시" for h in range(24)]

                fig = px.imshow(
                    heatmap_pivot.values,
                    labels=dict(x="시간", y="요일", color="CPU %"),
                    x=hour_labels,
                    y=day_labels,
                    color_continuous_scale='RdYlBu_r',
                    title="요일 × 시간대 CPU 사용률 히트맵",
                    aspect='auto'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')

            st.markdown("---")

            # 일별 추이
            st.subheader("📈 일별 추이")

            daily_trend = df_cpu.groupby('DATE')['CPU_USAGE_AVG'].agg(['mean', 'max', 'std']).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_trend['DATE'], y=daily_trend['max'],
                mode='lines', name='최대', line=dict(color='#e74c3c', width=1, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=daily_trend['DATE'], y=daily_trend['mean'],
                mode='lines+markers', name='평균', line=dict(color='#3498db', width=2)
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", annotation_text="경고")
            fig.update_layout(
                title="일별 CPU 사용률 추이 (전체)",
                xaxis_title="날짜",
                yaxis_title="CPU 사용률 (%)",
                height=300
            )
            st.plotly_chart(fig, width='stretch')

            # 서버별 시계열
            st.subheader("🖥️ 서버별 시계열 비교")

            # 서버별 일별 평균
            server_daily = df_cpu.groupby(['DATE', 'MNG_NO'])['CPU_USAGE_AVG'].mean().reset_index()

            fig = px.line(
                server_daily,
                x='DATE',
                y='CPU_USAGE_AVG',
                color='MNG_NO',
                title="서버별 일별 CPU 사용률",
                labels={'DATE': '날짜', 'CPU_USAGE_AVG': 'CPU 사용률 (%)', 'MNG_NO': '서버'},
                markers=True
            )
            fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", line_width=1)
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')

            # 서버별 시간대별 패턴
            server_hourly = df_cpu.groupby(['HOUR', 'MNG_NO'])['CPU_USAGE_AVG'].mean().reset_index()

            fig = px.line(
                server_hourly,
                x='HOUR',
                y='CPU_USAGE_AVG',
                color='MNG_NO',
                title="서버별 시간대별 CPU 사용률 패턴",
                labels={'HOUR': '시간', 'CPU_USAGE_AVG': 'CPU 사용률 (%)', 'MNG_NO': '서버'},
                markers=True
            )
            fig.add_hline(y=80, line_dash="dash", line_color="#f39c12", line_width=1)
            fig.update_layout(height=350, xaxis=dict(tickmode='linear', dtick=2))
            st.plotly_chart(fig, width='stretch')

            st.markdown("---")

            # 피크 시간대 분석
            st.subheader("⏰ 피크 시간대 분석")

            col1, col2, col3 = st.columns(3)

            hourly_mean = df_cpu.groupby('HOUR')['CPU_USAGE_AVG'].mean()
            peak_hour = hourly_mean.idxmax()
            low_hour = hourly_mean.idxmin()

            with col1:
                st.metric("피크 시간", f"{peak_hour}:00", f"평균 {hourly_mean[peak_hour]:.1f}%")
            with col2:
                st.metric("최저 시간", f"{low_hour}:00", f"평균 {hourly_mean[low_hour]:.1f}%")
            with col3:
                peak_low_diff = hourly_mean[peak_hour] - hourly_mean[low_hour]
                st.metric("피크-최저 차이", f"{peak_low_diff:.1f}%p")

            # 업무시간 vs 비업무시간
            st.markdown("#### 업무시간 vs 비업무시간")
            df_cpu['IS_WORK_HOUR'] = df_cpu['HOUR'].between(9, 18)

            work_stats = df_cpu.groupby('IS_WORK_HOUR')['CPU_USAGE_AVG'].agg(['mean', 'std', 'max'])
            work_stats.index = ['비업무시간 (19-08시)', '업무시간 (09-18시)']

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(work_stats.round(2).rename(columns={'mean': '평균', 'std': '표준편차', 'max': '최대'}),
                            width='stretch')
            with col2:
                fig = px.bar(
                    x=['업무시간', '비업무시간'],
                    y=[work_stats.loc['업무시간 (09-18시)', 'mean'],
                       work_stats.loc['비업무시간 (19-08시)', 'mean']],
                    color=['업무시간', '비업무시간'],
                    color_discrete_map={'업무시간': '#3498db', '비업무시간': '#95a5a6'},
                    title="업무/비업무 시간대 평균 CPU"
                )
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, width='stretch')

    # -------------- 탭5: 이상치 분석 --------------
    with tab5:
        st.header("🔬 이상치 분석")
        st.markdown("*각 알고리즘별 이상치 탐지 결과 미리보기 및 분포 특성 파악*")

        if 'sms_cpu' in data and 'CPU_USAGE_AVG' in data['sms_cpu'].columns:
            df_cpu = data['sms_cpu'].copy()

            # ===== 분포 특성 =====
            st.subheader("📊 분포 특성 분석")

            cpu_data = df_cpu['CPU_USAGE_AVG'].dropna()

            col1, col2, col3, col4 = st.columns(4)

            # 왜도 (Skewness)
            skewness = cpu_data.skew()
            with col1:
                skew_interp = "오른쪽 꼬리" if skewness > 0.5 else "왼쪽 꼬리" if skewness < -0.5 else "대칭"
                st.metric("왜도 (Skewness)", f"{skewness:.3f}", skew_interp)

            # 첨도 (Kurtosis)
            kurtosis = cpu_data.kurtosis()
            with col2:
                kurt_interp = "뾰족함 (이상치↑)" if kurtosis > 1 else "평평함" if kurtosis < -1 else "정규분포 유사"
                st.metric("첨도 (Kurtosis)", f"{kurtosis:.3f}", kurt_interp)

            # 변동계수 (CV)
            cv = (cpu_data.std() / cpu_data.mean()) * 100
            with col3:
                cv_interp = "높은 변동성" if cv > 50 else "중간 변동성" if cv > 25 else "낮은 변동성"
                st.metric("변동계수 (CV)", f"{cv:.1f}%", cv_interp)

            # 범위
            data_range = cpu_data.max() - cpu_data.min()
            with col4:
                st.metric("데이터 범위", f"{data_range:.1f}%p",
                         f"{cpu_data.min():.1f}% ~ {cpu_data.max():.1f}%")

            # 분포 시각화 + 정규분포 비교
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=cpu_data, nbinsx=50, name='실제 분포',
                    marker_color='#3498db', opacity=0.7,
                    histnorm='probability density'
                ))

                # 정규분포 곡선
                x_range = np.linspace(cpu_data.min(), cpu_data.max(), 100)
                normal_curve = stats.norm.pdf(x_range, cpu_data.mean(), cpu_data.std())
                fig.add_trace(go.Scatter(
                    x=x_range, y=normal_curve, mode='lines',
                    name='정규분포', line=dict(color='#e74c3c', width=2, dash='dash')
                ))

                fig.update_layout(
                    title="CPU 사용률 분포 vs 정규분포",
                    xaxis_title="CPU 사용률 (%)",
                    yaxis_title="확률 밀도",
                    height=350
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Q-Q Plot
                sorted_data = np.sort(cpu_data.sample(min(1000, len(cpu_data)), random_state=42))
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=theoretical_quantiles, y=sorted_data,
                    mode='markers', name='데이터',
                    marker=dict(color='#3498db', size=5, opacity=0.6)
                ))

                # 대각선
                min_val = min(theoretical_quantiles.min(), sorted_data.min())
                max_val = max(theoretical_quantiles.max(), sorted_data.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[cpu_data.mean() + cpu_data.std() * min_val,
                       cpu_data.mean() + cpu_data.std() * max_val],
                    mode='lines', name='이론적 정규분포',
                    line=dict(color='#e74c3c', dash='dash')
                ))

                fig.update_layout(
                    title="Q-Q Plot (정규성 검정)",
                    xaxis_title="이론적 분위수",
                    yaxis_title="실제 분위수",
                    height=350
                )
                st.plotly_chart(fig, width='stretch')

            st.markdown("---")

            # ===== 이상치 탐지 미리보기 =====
            st.subheader("🎯 이상치 탐지 미리보기")

            col1, col2 = st.columns(2)

            with col1:
                # Z-Score 분포
                z_scores = np.abs(stats.zscore(cpu_data))

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=z_scores, nbinsx=50, name='Z-Score 분포',
                    marker_color='#9b59b6'
                ))
                fig.add_vline(x=2, line_dash="dash", line_color="#f39c12",
                             annotation_text="2σ")
                fig.add_vline(x=3, line_dash="dash", line_color="#e74c3c",
                             annotation_text="3σ")
                fig.update_layout(
                    title="Z-Score 분포",
                    xaxis_title="|Z-Score|",
                    yaxis_title="빈도",
                    height=300
                )
                st.plotly_chart(fig, width='stretch')

                # Z-Score 이상치 비율
                z2_pct = (z_scores > 2).mean() * 100
                z3_pct = (z_scores > 3).mean() * 100
                st.markdown(f"- **2σ 초과**: {z2_pct:.2f}% ({(z_scores > 2).sum():,}건)")
                st.markdown(f"- **3σ 초과**: {z3_pct:.2f}% ({(z_scores > 3).sum():,}건)")

            with col2:
                # IQR 기반 이상치
                Q1 = cpu_data.quantile(0.25)
                Q3 = cpu_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=cpu_data.sample(min(5000, len(cpu_data)), random_state=42),
                    name='CPU 사용률',
                    marker_color='#1abc9c',
                    boxpoints='outliers'
                ))
                fig.update_layout(
                    title="IQR 기반 이상치 (박스플롯)",
                    yaxis_title="CPU 사용률 (%)",
                    height=300
                )
                st.plotly_chart(fig, width='stretch')

                iqr_outliers = ((cpu_data < lower_bound) | (cpu_data > upper_bound)).sum()
                iqr_pct = iqr_outliers / len(cpu_data) * 100
                st.markdown(f"- **IQR 경계**: [{lower_bound:.1f}%, {upper_bound:.1f}%]")
                st.markdown(f"- **이상치**: {iqr_pct:.2f}% ({iqr_outliers:,}건)")

            st.markdown("---")

            # ===== 서버별 안정성 =====
            st.subheader("🖥️ 서버별 안정성 분석")

            server_stats = df_cpu.groupby('MNG_NO')['CPU_USAGE_AVG'].agg([
                'mean', 'std', 'max',
                lambda x: (x >= 80).mean() * 100,  # 경고 비율
                lambda x: (x >= 90).mean() * 100   # 위험 비율
            ]).reset_index()
            server_stats.columns = ['서버', '평균', '표준편차', '최대', '경고비율(%)', '위험비율(%)']
            server_stats['변동계수'] = (server_stats['표준편차'] / server_stats['평균'] * 100).round(1)
            server_stats = server_stats.sort_values('변동계수', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                # 서버별 변동성 (표준편차)
                fig = px.bar(
                    server_stats,
                    x='서버', y='변동계수',
                    title="서버별 변동계수 (CV) - 높을수록 불안정",
                    color='변동계수',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig, width='stretch')

            with col2:
                # 서버별 위험 비율
                fig = px.bar(
                    server_stats,
                    x='서버', y='위험비율(%)',
                    title="서버별 위험 수준(90%↑) 비율",
                    color='위험비율(%)',
                    color_continuous_scale='OrRd'
                )
                fig.update_layout(height=300, coloraxis_showscale=False)
                st.plotly_chart(fig, width='stretch')

            # 서버별 상세 테이블
            st.markdown("#### 서버별 상세 통계")
            st.dataframe(
                server_stats.style.background_gradient(subset=['변동계수', '위험비율(%)'], cmap='Reds'),
                width='stretch'
            )

            # 고위험 서버 알림
            high_risk_servers = server_stats[server_stats['위험비율(%)'] > 5]
            if len(high_risk_servers) > 0:
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>⚠️ 고위험 서버 감지</strong><br>
                    위험 수준(90%↑) 비율이 5%를 초과하는 서버: <strong>{', '.join(high_risk_servers['서버'].astype(str))}</strong>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ===== 알고리즘 추천 =====
            st.subheader("💡 이상 탐지 알고리즘 추천")

            recommendations = []

            # 분포 기반 추천
            if abs(skewness) < 0.5 and abs(kurtosis) < 1:
                recommendations.append(("✅ Z-Score", "분포가 정규분포에 가까워 Z-Score가 효과적입니다."))
            else:
                recommendations.append(("⚠️ Z-Score", f"왜도({skewness:.2f}), 첨도({kurtosis:.2f})로 인해 정확도가 떨어질 수 있습니다."))

            recommendations.append(("✅ IQR", "분포 가정이 없어 비대칭 분포에도 강건합니다."))

            if len(server_stats) > 1:
                recommendations.append(("✅ Isolation Forest", "다변량(CPU+메모리) 분석에 적합합니다."))

            recommendations.append(("✅ 임계값 기반", "운영 기준(80%/90%)이 명확하여 해석이 용이합니다."))

            for algo, desc in recommendations:
                st.markdown(f"- **{algo}**: {desc}")

            st.markdown("---")

            # ===== 데이터 품질 분석 =====
            st.subheader("🔍 데이터 품질 분석")

            # 전체 데이터셋 품질 요약
            st.markdown("#### 📋 데이터셋별 결측치 현황")

            quality_data = []

            for name, key in [
                ('NMS 장비 마스터', 'nms_master'),
                ('NMS IF 성능', 'nms_if_perf'),
                ('SMS 서버 마스터', 'sms_master'),
                ('SMS CPU', 'sms_cpu'),
                ('SMS 메모리', 'sms_memory'),
                ('SMS 파일시스템', 'sms_filesystem')
            ]:
                if key in data and data[key] is not None:
                    df_temp = data[key]
                    total_cells = df_temp.shape[0] * df_temp.shape[1]
                    null_cells = df_temp.isnull().sum().sum()
                    null_pct = (null_cells / total_cells * 100) if total_cells > 0 else 0

                    quality_data.append({
                        '데이터셋': name,
                        '행 수': f"{len(df_temp):,}",
                        '컬럼 수': df_temp.shape[1],
                        '결측 셀': f"{null_cells:,}",
                        '결측률(%)': f"{null_pct:.2f}"
                    })

            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df, width='stretch')

            # SMS CPU 상세 분석
            st.markdown("#### 💻 SMS CPU 데이터 상세 품질")

            col1, col2 = st.columns(2)

            with col1:
                # 컬럼별 결측치
                null_counts = df_cpu.isnull().sum()
                null_cols = null_counts[null_counts > 0]

                if len(null_cols) > 0:
                    fig = px.bar(
                        x=null_cols.index,
                        y=null_cols.values,
                        title="컬럼별 결측치 수",
                        labels={'x': '컬럼', 'y': '결측치 수'},
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.success("✅ CPU 데이터에 결측치가 없습니다.")

            with col2:
                # 데이터 수집 현황 (서버 × 날짜)
                if 'YYYYMMDD' in df_cpu.columns:
                    collection_matrix = df_cpu.groupby(['MNG_NO', 'YYYYMMDD']).size().unstack(fill_value=0)

                    fig = px.imshow(
                        collection_matrix.values,
                        labels=dict(x="날짜", y="서버", color="수집 건수"),
                        x=[str(c) for c in collection_matrix.columns],
                        y=[str(i) for i in collection_matrix.index],
                        color_continuous_scale='Blues',
                        title="서버 × 날짜별 데이터 수집 현황",
                        aspect='auto'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')

            # 이상 값 탐지 (0%, 100% 등 비정상 값)
            st.markdown("#### ⚠️ 비정상 값 탐지")

            col1, col2, col3 = st.columns(3)

            with col1:
                zero_count = (df_cpu['CPU_USAGE_AVG'] == 0).sum()
                st.metric("CPU 0% 건수", f"{zero_count:,}건",
                         f"{zero_count/len(df_cpu)*100:.2f}%" if len(df_cpu) > 0 else "0%")

            with col2:
                full_count = (df_cpu['CPU_USAGE_AVG'] >= 100).sum()
                st.metric("CPU 100%↑ 건수", f"{full_count:,}건",
                         f"{full_count/len(df_cpu)*100:.2f}%" if len(df_cpu) > 0 else "0%")

            with col3:
                negative_count = (df_cpu['CPU_USAGE_AVG'] < 0).sum()
                st.metric("CPU 음수 건수", f"{negative_count:,}건",
                         "데이터 오류" if negative_count > 0 else "정상",
                         delta_color="inverse" if negative_count > 0 else "normal")

            # 마스터 vs 성능 데이터 비교
            st.markdown("#### 🔗 마스터 vs 성능 데이터 매칭")

            if 'sms_master' in data:
                master_servers = set(data['sms_master']['MNG_NO'].unique())
                perf_servers = set(df_cpu['MNG_NO'].unique())

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("마스터 등록 서버", f"{len(master_servers)}대")
                with col2:
                    st.metric("성능 데이터 서버", f"{len(perf_servers)}대")
                with col3:
                    missing = master_servers - perf_servers
                    st.metric("데이터 누락 서버", f"{len(missing)}대",
                             "확인 필요" if len(missing) > 0 else "정상",
                             delta_color="inverse" if len(missing) > 0 else "normal")

                if len(missing) > 0 and len(missing) <= 10:
                    st.warning(f"⚠️ 성능 데이터 누락 서버: {sorted(missing)}")
                elif len(missing) > 10:
                    st.warning(f"⚠️ 성능 데이터 누락 서버: {len(missing)}대 (마스터에 등록되었으나 CPU 데이터 없음)")


# ============== 메인 대시보드 ==============
def main():
    # 데이터 로딩
    with st.spinner("데이터 로딩 중..."):
        data = load_data()

    if data is None:
        st.error("데이터를 로딩할 수 없습니다. output 폴더에 CSV 파일이 있는지 확인해주세요.")
        return

    # 사이드바 - 페이지 선택
    st.sidebar.title("🔍 NETIS")
    page = st.sidebar.radio(
        "페이지 선택",
        ["📁 원본 데이터 EDA", "📊 통합 데이터 분석", "🎯 이상치 탐지 가이드", "🖥️ SMS 이상 탐지", "🌐 NMS 이상 탐지"]
    )

    # 원본 데이터 EDA 페이지
    if page == "📁 원본 데이터 EDA":
        page_raw_eda()
        return

    # 통합 EDA 페이지
    if page == "📊 통합 데이터 분석":
        page_eda(data)
        return

    # 이상치 탐지 가이드 페이지
    if page == "🎯 이상치 탐지 가이드":
        page_anomaly_guide()
        return

    # 이상 탐지 페이지 공통 설정
    st.title("🔍 NETIS 이상 탐지 시스템")
    st.markdown("**SFR-002: 장애 발생 전 징후 포착**")

    # 사이드바 - 이상 탐지 파라미터
    st.sidebar.header("⚙️ 설정")
    st.sidebar.subheader("이상 탐지 파라미터")
    z_threshold = st.sidebar.slider("Z-Score 임계값", 2.0, 4.0, 3.0, 0.5)
    warning_threshold = st.sidebar.slider("경고 임계값 (%)", 60, 90, 80, 5)
    critical_threshold = st.sidebar.slider("위험 임계값 (%)", 70, 99, 90, 5)

    # LLM 설정 (내부망 Ollama)
    st.sidebar.subheader("LLM 설정")
    st.sidebar.caption(f"서버: {LLMAnalyzer.OLLAMA_HOST}")
    llm_model = st.sidebar.text_input("Ollama 모델", LLMAnalyzer.OLLAMA_MODEL)

    # ============== SMS 분석 ==============
    if page == "🖥️ SMS 이상 탐지":
        st.header("📊 SMS 서버 이상 탐지")

        if 'sms_cpu' not in data or 'CPU_USAGE_AVG' not in data['sms_cpu'].columns:
            st.warning("SMS CPU 데이터가 없습니다.")
            return

        df_cpu = data['sms_cpu'].copy()
        df_memory = data['sms_memory'].copy()

        # 서버 선택
        server_list = df_cpu['MNG_NO'].unique()
        selected_server = st.selectbox("서버 선택", ["전체"] + list(server_list))

        if selected_server != "전체":
            df_cpu = df_cpu[df_cpu['MNG_NO'] == selected_server]
            df_memory = df_memory[df_memory['MNG_NO'] == selected_server]

        # 이상 탐지 실행
        detector = AnomalyDetector()

        # CPU 이상 탐지
        df_cpu['is_anomaly_zscore'] = detector.detect_zscore(df_cpu['CPU_USAGE_AVG'], z_threshold)
        df_cpu['severity'] = detector.detect_threshold(df_cpu['CPU_USAGE_AVG'], warning_threshold, critical_threshold)
        df_cpu['is_anomaly'] = df_cpu['is_anomaly_zscore'] | (df_cpu['severity'] != 'normal')

        # 메모리 이상 탐지
        if 'PHYSICAL_USED_PCT' in df_memory.columns:
            df_memory['is_anomaly_zscore'] = detector.detect_zscore(df_memory['PHYSICAL_USED_PCT'], z_threshold)
            df_memory['severity'] = detector.detect_threshold(df_memory['PHYSICAL_USED_PCT'], warning_threshold, critical_threshold)
            df_memory['is_anomaly'] = df_memory['is_anomaly_zscore'] | (df_memory['severity'] != 'normal')

        # ============== 메트릭 카드 ==============
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_records = len(df_cpu)
            st.metric("총 데이터", f"{total_records:,}건")

        with col2:
            anomaly_count = df_cpu['is_anomaly'].sum()
            st.metric("이상 탐지", f"{anomaly_count:,}건",
                     delta=f"{anomaly_count/total_records*100:.1f}%")

        with col3:
            critical_count = (df_cpu['severity'] == 'critical').sum()
            st.metric("위험 수준", f"{critical_count:,}건", delta_color="inverse")

        with col4:
            warning_count = (df_cpu['severity'] == 'warning').sum()
            st.metric("경고 수준", f"{warning_count:,}건", delta_color="inverse")

        # ============== 시계열 차트 ==============
        st.subheader("📈 CPU 사용률 추이")

        fig = go.Figure()

        # 정상 데이터
        normal_data = df_cpu[~df_cpu['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['DATETIME'],
            y=normal_data['CPU_USAGE_AVG'],
            mode='lines',
            name='정상',
            line=dict(color='blue', width=1),
            opacity=0.7
        ))

        # 이상 데이터
        anomaly_data = df_cpu[df_cpu['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomaly_data['DATETIME'],
            y=anomaly_data['CPU_USAGE_AVG'],
            mode='markers',
            name='이상',
            marker=dict(color='red', size=8, symbol='x'),
        ))

        # 임계값 선
        fig.add_hline(y=warning_threshold, line_dash="dash", line_color="orange",
                     annotation_text=f"경고 ({warning_threshold}%)")
        fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"위험 ({critical_threshold}%)")

        fig.update_layout(
            xaxis_title="시간",
            yaxis_title="CPU 사용률 (%)",
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, width='stretch')

        # ============== CPU vs 메모리 분포 ==============
        if 'PHYSICAL_USED_PCT' in df_memory.columns:
            st.subheader("📊 CPU vs 메모리 상관관계")

            # 데이터 병합
            df_combined = df_cpu.merge(
                df_memory[['YYYYMMDD', 'TIME_ID', 'MNG_NO', 'PHYSICAL_USED_PCT']],
                on=['YYYYMMDD', 'TIME_ID', 'MNG_NO'],
                how='inner'
            )

            if len(df_combined) > 0:
                # Isolation Forest로 복합 이상 탐지
                is_anomaly_ml, scores = detector.detect_isolation_forest(
                    df_combined, ['CPU_USAGE_AVG', 'PHYSICAL_USED_PCT']
                )
                df_combined['is_anomaly_ml'] = is_anomaly_ml

                fig2 = px.scatter(
                    df_combined,
                    x='CPU_USAGE_AVG',
                    y='PHYSICAL_USED_PCT',
                    color='is_anomaly_ml',
                    color_discrete_map={True: 'red', False: 'blue'},
                    opacity=0.5,
                    labels={
                        'CPU_USAGE_AVG': 'CPU 사용률 (%)',
                        'PHYSICAL_USED_PCT': '메모리 사용률 (%)',
                        'is_anomaly_ml': '이상 여부'
                    }
                )

                fig2.add_vline(x=warning_threshold, line_dash="dash", line_color="orange")
                fig2.add_hline(y=warning_threshold, line_dash="dash", line_color="orange")

                fig2.update_layout(height=400)
                st.plotly_chart(fig2, width='stretch')

        # ============== 이상 탐지 상세 ==============
        st.subheader("🚨 이상 탐지 상세")

        if len(anomaly_data) > 0:
            # 서버 정보 병합
            anomaly_detail = anomaly_data.merge(
                data['sms_master'][['MNG_NO', 'DEV_NAME', 'DEV_IP']],
                on='MNG_NO',
                how='left'
            )

            display_cols = ['DATETIME', 'MNG_NO', 'DEV_NAME', 'DEV_IP', 'CPU_USAGE_AVG', 'severity']
            available_cols = [c for c in display_cols if c in anomaly_detail.columns]

            st.dataframe(
                anomaly_detail[available_cols].sort_values('DATETIME', ascending=False).head(50),
                width='stretch'
            )

            # ============== LLM 분석 (내부망 Ollama) ==============
            st.subheader("🤖 AI 분석")

            llm = LLMAnalyzer(model=llm_model)  # 내부망 서버 자동 연결

            if st.button("AI 분석 실행", type="primary"):
                if llm.check_connection():
                    # 프롬프트 생성
                    prompt = f"""
다음은 서버 모니터링 시스템에서 탐지된 이상 징후입니다:

## 탐지 개요
- 총 데이터: {total_records:,}건
- 이상 탐지: {anomaly_count:,}건 ({anomaly_count/total_records*100:.1f}%)
- 위험 수준: {critical_count:,}건
- 경고 수준: {warning_count:,}건

## 이상 서버 목록 (상위 5개)
"""
                    for _, row in anomaly_detail.head(5).iterrows():
                        prompt += f"- {row.get('DEV_NAME', 'Unknown')} (IP: {row.get('DEV_IP', 'Unknown')}): CPU {row['CPU_USAGE_AVG']:.1f}%\n"

                    prompt += """
## 분석 요청
1. 위 이상 징후의 가능한 원인을 분석해주세요.
2. 예상되는 영향 범위를 설명해주세요.
3. 권장하는 조치 사항을 우선순위와 함께 제시해주세요.
"""

                    with st.spinner("AI 분석 중..."):
                        result = llm.analyze(prompt)
                        if result:
                            st.markdown("### 분석 결과")
                            st.markdown(result)
                else:
                    st.warning(f"""
                    ⚠️ 내부망 Ollama 서버에 연결할 수 없습니다.

                    **서버 정보:**
                    - 주소: {llm.base_url}
                    - 모델: {llm.model}

                    **확인 사항:**
                    - 내부망(사내 네트워크)에서만 접속 가능합니다.
                    - 외부에서 접속하려면 ICT부서에 API 접속 권한을 요청하세요.
                    """)
        else:
            st.success("✅ 이상 징후가 탐지되지 않았습니다.")

    # ============== NMS 분석 ==============
    elif page == "🌐 NMS 이상 탐지":
        st.header("🌐 NMS 네트워크 이상 탐지")

        if 'nms_if_perf' not in data:
            st.warning("NMS IF 성능 데이터가 없습니다.")
            return

        df_if = data['nms_if_perf'].copy()

        # 샘플링 (데이터가 크므로)
        if len(df_if) > 50000:
            df_if = df_if.sample(n=50000, random_state=42)
            st.info(f"대용량 데이터로 인해 50,000건 샘플링 수행")

        # 장비 선택
        device_list = df_if['MNG_NO'].unique()
        selected_device = st.selectbox("장비 선택", ["전체"] + list(device_list[:50]))

        if selected_device != "전체":
            df_if = df_if[df_if['MNG_NO'] == selected_device]

        # 이상 탐지
        detector = AnomalyDetector()

        feature_cols = ['AVG_INBPS', 'AVG_OUTBPS']
        available_cols = [c for c in feature_cols if c in df_if.columns]

        if len(available_cols) >= 2:
            is_anomaly, scores = detector.detect_isolation_forest(df_if, available_cols)
            df_if['is_anomaly'] = is_anomaly
            df_if['anomaly_score'] = scores

            # 메트릭
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("총 데이터", f"{len(df_if):,}건")
            with col2:
                anomaly_count = df_if['is_anomaly'].sum()
                st.metric("이상 탐지", f"{anomaly_count:,}건")
            with col3:
                st.metric("이상 비율", f"{anomaly_count/len(df_if)*100:.2f}%")

            # 트래픽 시계열
            st.subheader("📈 트래픽 추이")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('인바운드 트래픽', '아웃바운드 트래픽'))

            # 인바운드
            normal = df_if[~df_if['is_anomaly']]
            anomaly = df_if[df_if['is_anomaly']]

            fig.add_trace(go.Scatter(x=normal['DATETIME'], y=normal['AVG_INBPS'],
                                    mode='markers', name='정상 (IN)', marker=dict(size=3, color='blue'),
                                    opacity=0.5), row=1, col=1)
            fig.add_trace(go.Scatter(x=anomaly['DATETIME'], y=anomaly['AVG_INBPS'],
                                    mode='markers', name='이상 (IN)', marker=dict(size=6, color='red')),
                         row=1, col=1)

            # 아웃바운드
            fig.add_trace(go.Scatter(x=normal['DATETIME'], y=normal['AVG_OUTBPS'],
                                    mode='markers', name='정상 (OUT)', marker=dict(size=3, color='green'),
                                    opacity=0.5), row=2, col=1)
            fig.add_trace(go.Scatter(x=anomaly['DATETIME'], y=anomaly['AVG_OUTBPS'],
                                    mode='markers', name='이상 (OUT)', marker=dict(size=6, color='red')),
                         row=2, col=1)

            fig.update_layout(height=600)
            st.plotly_chart(fig, width='stretch')

            # 이상 데이터 상세
            st.subheader("🚨 이상 트래픽 상세")

            if len(anomaly) > 0:
                # 장비 정보 병합
                anomaly_with_info = anomaly.merge(
                    data['nms_master'][['MNG_NO', 'DEV_NAME', 'DEV_IP']],
                    on='MNG_NO',
                    how='left'
                )

                anomaly_sorted = anomaly_with_info.nsmallest(20, 'anomaly_score')
                display_cols = ['DATETIME', 'MNG_NO', 'DEV_NAME', 'DEV_IP', 'IF_IDX', 'AVG_INBPS', 'AVG_OUTBPS', 'anomaly_score']
                available_display = [c for c in display_cols if c in anomaly_sorted.columns]
                st.dataframe(anomaly_sorted[available_display], width='stretch')

                # ============== LLM 분석 (NMS) ==============
                st.subheader("🤖 AI 분석")

                llm = LLMAnalyzer(model=llm_model)

                total_records = len(df_if)

                if st.button("AI 분석 실행", type="primary", key="nms_llm_btn"):
                    if llm.check_connection():
                        # 이상 트래픽 통계 계산
                        avg_in_anomaly = anomaly['AVG_INBPS'].mean() if 'AVG_INBPS' in anomaly.columns else 0
                        avg_out_anomaly = anomaly['AVG_OUTBPS'].mean() if 'AVG_OUTBPS' in anomaly.columns else 0
                        max_in_anomaly = anomaly['AVG_INBPS'].max() if 'AVG_INBPS' in anomaly.columns else 0
                        max_out_anomaly = anomaly['AVG_OUTBPS'].max() if 'AVG_OUTBPS' in anomaly.columns else 0

                        affected_devices = anomaly['MNG_NO'].nunique()

                        # 프롬프트 생성
                        prompt = f"""
다음은 네트워크 모니터링 시스템(NMS)에서 탐지된 트래픽 이상 징후입니다:

## 탐지 개요
- 분석 데이터: {total_records:,}건
- 이상 탐지: {anomaly_count:,}건 ({anomaly_count/total_records*100:.1f}%)
- 영향 받은 장비 수: {affected_devices}대
- 영향 받은 인터페이스 수: {len(anomaly)}개

## 이상 트래픽 통계
- 평균 인바운드: {avg_in_anomaly:,.0f} bps
- 평균 아웃바운드: {avg_out_anomaly:,.0f} bps
- 최대 인바운드: {max_in_anomaly:,.0f} bps
- 최대 아웃바운드: {max_out_anomaly:,.0f} bps

## 이상 장비 목록 (상위 10개)
"""
                        for _, row in anomaly_sorted.head(10).iterrows():
                            dev_name = row.get('DEV_NAME', 'Unknown')
                            dev_ip = row.get('DEV_IP', 'Unknown')
                            in_bps = row.get('AVG_INBPS', 0)
                            out_bps = row.get('AVG_OUTBPS', 0)
                            prompt += f"- {dev_name} (IP: {dev_ip}): IN {in_bps:,.0f} bps, OUT {out_bps:,.0f} bps\n"

                        prompt += """
## 분석 요청
1. 위 트래픽 이상의 가능한 원인을 분석해주세요. (예: DDoS, 대용량 전송, 장비 장애 등)
2. 네트워크에 미치는 영향을 설명해주세요.
3. 권장하는 조치 사항을 우선순위와 함께 제시해주세요.
4. 향후 유사 상황을 예방하기 위한 방안을 제안해주세요.
"""

                        with st.spinner("AI 분석 중... (최대 3분 소요)"):
                            result = llm.analyze(prompt)
                            if result:
                                st.markdown("### 분석 결과")
                                st.markdown(result)
                            else:
                                st.error("분석 결과를 받지 못했습니다.")
                    else:
                        st.warning(f"""
                        ⚠️ 내부망 Ollama 서버에 연결할 수 없습니다.

                        **서버 정보:**
                        - 주소: {llm.base_url}
                        - 모델: {llm.model}

                        **확인 사항:**
                        - 내부망(사내 네트워크)에서만 접속 가능합니다.
                        - 외부에서 접속하려면 ICT부서에 API 접속 권한을 요청하세요.
                        """)
            else:
                st.success("✅ 이상 트래픽이 탐지되지 않았습니다.")


if __name__ == "__main__":
    main()
