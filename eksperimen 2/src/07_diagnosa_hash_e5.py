from pathlib import Path
import hashlib,json,csv,math

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parent.parent.parent/'eksperimen5'
    d=root/'output/testing/11_final_holdout_test_v1'
    f=d/'11_test_comparison.csv'
    r=d/'11_final_holdout_test_report.json'
    print('EXPERIMEN 5 — HASH DIAGNOSTIC (READ ONLY)')
    print('Report:',r)
    print('CSV:   ',f)
    if not r.exists() or not f.exists():
        print('FAIL: Report atau CSV tidak ditemukan. Periksa lokasi folder.');return 2
    report=json.loads(r.read_text(encoding='utf-8'))
    expected=report['output_sha256']['11_test_comparison.csv']
    actual=sha(f)
    print('SHA256 EXPECTED:',expected)
    print('SHA256 ACTUAL:  ',actual)
    print('HASH STATUS:', 'MATCH' if actual==expected else 'MISMATCH')
    try:
        with f.open(encoding='utf-8-sig',newline='') as fd:
            rows=list(csv.DictReader(fd))
        original={str(i['model_code']):i for i in report['all_model_test_metrics']}
        if len(rows)!=4 or len(original)!=4 or {r['model_code'] for r in rows}!=set(original):
            print('METRIC STATUS: FAIL rows/model_code mismatch');return 1
        errors=[]
        cols=['auroc','accuracy','macro_f1','balanced_accuracy','sensitivity_dr','specificity_ds',
              'tn','fp','fn','tp','threshold']
        for row in rows:
            code=row['model_code'];j=original[code]
            for k in cols:
                if k in j and k in row:
                    try:
                        if not math.isclose(float(row[k]),float(j[k]),rel_tol=1e-10,abs_tol=1e-10):
                            errors.append(f'{code}/{k}: CSV={row[k]} LOCK={j[k]}')
                    except ValueError: errors.append(f'{code}/{k}: invalid value {row[k]!r}')
        if errors:
            print('METRIC STATUS: MISMATCH');print(*errors,sep='\n')
        else: print('METRIC STATUS: MATCH on AUROC, F1, accuracy, sensitivity/specificity, confusion counts.')
    except Exception as ex:
        print('METRIC STATUS: UNREADABLE',type(ex).__name__,str(ex));return 1
    if actual!=expected:
        print('INSTRUCTION: Jangan mengubah E5. Gunakan reference CSV terverifikasi pada paket patch jika diperlukan.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
