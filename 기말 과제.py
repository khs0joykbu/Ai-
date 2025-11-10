# timetable_gradio_prototype.py
# 사용법: pip install pandas gradio caas_jupyter_tools openpyxl
# 그리고: python timetable_gradio_prototype.py
    

import pandas as pd
import numpy as np
import os
import gradio as gr

# (1) 경로 수정: 업로드한 엑셀 파일 경로를 지정하세요.
xlsx_path = "/mnt/data/20250915_학생별수강신청조회(학생).xlsx"

# (2) 엑셀 로드 및 컬럼 자동 추론
xls = pd.ExcelFile(xlsx_path)
sheets = xls.sheet_names
df = pd.read_excel(xlsx_path, sheet_name=sheets[0])

possible_cols = { 'student_id': None, 'student_name': None, 'course_code': None, 'course_name': None,
                  'credits': None, 'hours': None, 'instructor': None, 'lab_required': None, 'year': None, 'major': None }

for c in df.columns:
    lc = c.lower()
    if '학번' in lc or 'student' in lc or lc=='id':
        possible_cols['student_id'] = c
    if '이름' in lc or 'name' in lc:
        possible_cols['student_name'] = c
    if '과목' in lc or 'course' in lc:
        if '코드' in lc or 'code' in lc:
            possible_cols['course_code'] = c
        else:
            if possible_cols['course_name'] is None:
                possible_cols['course_name'] = c
    if '시수' in lc or '시간' in lc or 'hour' in lc or 'credit' in lc or '학점' in lc:
        possible_cols['hours'] = c
    if '교수' in lc or '담당' in lc or 'instructor' in lc:
        possible_cols['instructor'] = c
    if '실습' in lc or 'lab' in lc:
        possible_cols['lab_required'] = c

text_cols = [c for c in df.columns if df[c].dtype == object]
if possible_cols['course_name'] is None and len(text_cols)>0:
    possible_cols['course_name'] = text_cols[0]
if possible_cols['student_name'] is None and len(text_cols)>1:
    possible_cols['student_name'] = text_cols[1]

# (3) 샘플/기본 강의실(실제 환경에 맞춰 수정하세요)
rooms = pd.DataFrame([
    {"room_id":"L101","capacity":30,"is_lab":True},
    {"room_id":"L102","capacity":24,"is_lab":True},
    {"room_id":"R201","capacity":60,"is_lab":False},
    {"room_id":"R202","capacity":40,"is_lab":False},
    {"room_id":"R203","capacity":40,"is_lab":False},
])

days = ["Mon","Tue","Wed","Thu","Fri"]
hours = list(range(9,18))  # 9..17
timeslots = [f"{d} {h}:00-{h+1}:00" for d in days for h in hours]

# (4) 과목 수요 집계
if possible_cols['course_name'] in df.columns:
    course_groups = df.groupby(possible_cols['course_name']).agg(
        enrolled_students = (possible_cols['student_id'] if possible_cols['student_id'] in df.columns else df.columns[0], 'nunique')
    ).reset_index().rename(columns={possible_cols['course_name']:'course_name'})
else:
    course_groups = df.groupby(df.columns[0]).size().reset_index().rename(columns={0:'enrolled_students', df.columns[0]:'course_name'})

def infer_hours(row):
    hcol = possible_cols['hours']
    if hcol and hcol in df.columns:
        vals = df[df[possible_cols['course_name']]==row['course_name']][hcol].dropna().unique()
        if len(vals)>0:
            try:
                return int(vals[0])
            except:
                try:
                    return int(float(vals[0]))
                except:
                    return 2
    return 2

def infer_lab(row):
    lcol = possible_cols['lab_required']
    if lcol and lcol in df.columns:
        vals = df[df[possible_cols['course_name']]==row['course_name']][lcol].dropna().unique()
        if len(vals)>0:
            v = str(vals[0]).strip().lower()
            if v in ['y','yes','true','t','실습','있음','1']:
                return True
            if '실습' in v or 'lab' in v:
                return True
    if '실습' in str(row['course_name']):
        return True
    return False

catalog = course_groups.copy()
catalog['hours_per_week'] = catalog.apply(infer_hours, axis=1)
catalog['lab_required'] = catalog.apply(infer_lab, axis=1)
catalog['expected_capacity'] = catalog['enrolled_students']

# (5) 간단 그리디 스케줄러
def run_scheduler(catalog_df, rooms_df, timeslots_list):
    availability = {(r['room_id'], ts): True for _,r in rooms_df.iterrows() for ts in timeslots_list}
    assignments = []
    cat = catalog_df.sort_values(['expected_capacity','hours_per_week'], ascending=[False,False]).copy()
    for _, row in cat.iterrows():
        need = int(row['hours_per_week'])
        placed = 0
        for ts in timeslots_list:
            if placed>=need:
                break
            for _, room in rooms_df.iterrows():
                if row['lab_required'] and (not room['is_lab']):
                    continue
                if room['capacity'] < row['expected_capacity'] and room['capacity'] < 20:
                    continue
                if availability[(room['room_id'], ts)]:
                    assignments.append({
                        "course_name": row['course_name'],
                        "room_id": room['room_id'],
                        "timeslot": ts,
                        "hours": 1
                    })
                    availability[(room['room_id'], ts)] = False
                    placed += 1
                    break
        if placed < need:
            assignments.append({
                "course_name": row['course_name'],
                "room_id": None,
                "timeslot": None,
                "hours": need-placed,
                "status":"unplaced"
            })
    assign_df = pd.DataFrame(assignments)
    return assign_df, availability

# (6) UI 함수
out_csv = "schedule_result.csv"

def schedule_and_view(dummy=1):
    assign_df2, availability2 = run_scheduler(catalog, rooms, timeslots)
    csv_text = assign_df2.to_csv(index=False)
    placed = assign_df2['room_id'].notna().sum()
    total_slots = assign_df2.shape[0]
    unplaced = assign_df2[assign_df2['room_id'].isna()]
    summary = f"Assigned slots: {placed}\\nTotal slot-rows: {total_slots}\\nUnplaced course-hours rows: {unplaced.shape[0]}"
    assign_df2.to_csv(out_csv, index=False)
    return csv_text, summary

# (7) Gradio 인터페이스 실행
iface = gr.Interface(fn=schedule_and_view, inputs=gr.inputs.Number(default=1, label="Run (enter any number)"),
                     outputs=[gr.outputs.Textbox(label="배정 결과 CSV"), gr.outputs.Textbox(label="요약")],
                     title="시간표 자동 배정 데모 (프로토타입)",
                     description="- 간단한 그리디 스케줄러 예제입니다. 실제로는 추가 제약조건이 필요합니다.")
iface.launch(share=False, inbrowser=True)
