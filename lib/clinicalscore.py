import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch


class SOFA:

    """
    SOFA score is used to determine sepsis. it is made up of six systematic scores.
    """

    @ staticmethod
    def respiration(po2_fio2, vent, point_po2_fio2=(400, 300, 200, 100)):
        # po2_fio2_last, vent_last
        if pd.isna(po2_fio2):
            grade_res = pd.NA
        elif po2_fio2 >= point_po2_fio2[0]:
            grade_res = 0
        elif po2_fio2 >= point_po2_fio2[1]:
            grade_res = 1
        elif po2_fio2 >= point_po2_fio2[2]:
            grade_res = 2
        elif po2_fio2 >= point_po2_fio2[3]:
            grade_res = 3
        else:
            grade_res = 4
            if pd.isna(vent):
                grade_res = 3
            elif vent == 0:
                grade_res = 3
        return grade_res

    @ staticmethod
    def coagulation(platelet, point_platelet=(150, 100, 50, 20)):
        # platelet_min
        if pd.isna(platelet):
            grade_coa = pd.NA
        elif platelet >= point_platelet[0]:
            grade_coa = 0
        elif platelet >= point_platelet[1]:
            grade_coa = 1
        elif platelet >= point_platelet[2]:
            grade_coa = 2
        elif platelet >= point_platelet[3]:
            grade_coa = 3
        else:
            grade_coa = 4
        return grade_coa

    @ staticmethod
    def liver(bilirubin_total, point_bilirubin_total=(12, 6, 2, 1.2)):
        # bilirubin_total_max
        if pd.isna(bilirubin_total):
            grade_liv = pd.NA
        elif bilirubin_total >= point_bilirubin_total[0]:
            grade_liv = 4
        elif bilirubin_total >= point_bilirubin_total[1]:
            grade_liv = 3
        elif bilirubin_total >= point_bilirubin_total[2]:
            grade_liv = 2
        elif bilirubin_total >= point_bilirubin_total[3]:
            grade_liv = 1
        else:
            grade_liv = 0
        return grade_liv

    @ staticmethod
    def cardiovascular(mbp, dopamine, dobutamine, epinephrine, norepinephrine,
                       point_mbp=70,
                       point_dopamine=(5, 15),
                       point_dobutamine=0,
                       point_epinephrine=0.1,
                       point_norepinephrine=0.1):
        # mbp_min, dopamine_max, dobutamine_last, epinephrine_max, norepinephrine_max
        if pd.isna(mbp):
            grade_mbp = pd.NA
        elif mbp >= point_mbp:
            grade_mbp = 0
        else:
            grade_mbp = 1

        if pd.isna(dopamine):
            grade_dopamine = pd.NA
        elif dopamine <= point_dopamine[0]:
            grade_dopamine = 2
        elif dopamine <= point_dopamine[1]:
            grade_dopamine = 3
        else:
            grade_dopamine = 4

        if pd.isna(dobutamine):
            grade_dobutamine = pd.NA
        elif dobutamine > point_dobutamine:
            grade_dobutamine = 2
        else:
            grade_dobutamine = 2

        if pd.isna(epinephrine):
            grade_epinephrine = pd.NA
        elif epinephrine <= point_epinephrine:
            grade_epinephrine = 3
        else:
            grade_epinephrine = 4

        if pd.isna(norepinephrine):
            grade_norepinephrine = pd.NA
        elif norepinephrine <= point_norepinephrine:
            grade_norepinephrine = 3
        else:
            grade_norepinephrine = 4

        grade_car = pd.DataFrame([grade_mbp, grade_dopamine, grade_dobutamine,
                                  grade_epinephrine, grade_norepinephrine]).max()[0]
        if grade_car is None:
            grade_car = pd.NA
        return grade_car

    @ staticmethod
    def neurological(gcs, point_gcs=(14, 13, 12, 10, 9, 6)):
        # gcs_min
        if pd.isna(gcs):
            grade_neu = pd.NA
        elif point_gcs[0] >= gcs >= point_gcs[1]:
            grade_neu = 1
        elif point_gcs[2] >= gcs >= point_gcs[3]:
            grade_neu = 2
        elif point_gcs[4] >= gcs >= point_gcs[5]:
            grade_neu = 3
        elif gcs < point_gcs[5]:
            grade_neu = 4
        else:
            grade_neu = 0
        return grade_neu

    @ staticmethod
    def renal(creatinine, outputtoal, point_creatinine=(5.0, 3.5, 2.0, 1.2), point_outputtoal=(200, 500)):
        # creatinine_max, outputtoal_sum
        if pd.isna(creatinine):
            grade_creatinine = pd.NA
        elif creatinine >= point_creatinine[0]:
            grade_creatinine = 4
        elif creatinine >= point_creatinine[1]:
            grade_creatinine = 3
        elif creatinine >= point_creatinine[2]:
            grade_creatinine = 2
        elif creatinine >= point_creatinine[3]:
            grade_creatinine = 1
        else:
            grade_creatinine = 0

        if pd.isna(outputtoal):
            grade_outputtoal = pd.NA
        elif outputtoal < point_outputtoal[0]:
            grade_outputtoal = 4
        elif outputtoal < point_outputtoal[1]:
            grade_outputtoal = 3
        else:
            grade_outputtoal = 0

        # if pd.isna(grade_creatinine):
        #     if pd.isna(grade_outputtoal):
        #         grade_ren = pd.NA
        #     else:
        #         grade_ren = grade_outputtoal
        # else:
        #     if pd.isna(grade_outputtoal):
        #         grade_ren = grade_creatinine
        #     else:
        #         grade_ren = max(grade_creatinine, grade_outputtoal)

        grade_ren = pd.DataFrame([grade_creatinine, grade_outputtoal]).max()[0]
        if grade_ren is None:
            grade_ren = pd.NA
        return grade_ren

    @ staticmethod
    def get_sofa_score(values, thres=0.5):
        """
        values: ['po2_fio2_last', 'vent_last', 'platelet_min', 'bilirubin_total_max', 'mbp_min', 'dopamine_max',
                 'dobutamine_last', 'epinephrine_max', 'norepinephrine_max', 'gcs_min', 'creatinine_max',
                 'outputtoal_sum']
        thres: the rate of missing sub score of sofa.
        """
        sub_score = pd.DataFrame([SOFA.respiration(values[0], values[1]),
                                  SOFA.coagulation(values[2]),
                                  SOFA.liver(values[3]),
                                  SOFA.cardiovascular(values[4], values[5], values[6], values[7], values[8]),
                                  SOFA.neurological(values[9]),
                                  SOFA.renal(values[10], values[11])])

        if np.sum(pd.isna(sub_score[0])) / len(sub_score) > thres:
            sofa = pd.NA
        else:
            sofa = sub_score.sum()[0]
        return sub_score[0].values, sofa


class ClinicalScore:
    """
    This is clinical score which is related to disease "Sepsis".

    Including:
        1 SOFA
        2 qSOFA
        3 mLODS
        4 SIRS
        5 NEWS
    """

    def __init__(self, args):
        self.args = args

    def sofa_score(self, values):
        columns_sofa = ['respiration', 'coagulation', 'liver', 'cardiovascular', 'neurological', 'renal', 'sofa']
        sofa_all_time = []
        for val in values:
            # print(val.shape)
            sub_score, sofa = SOFA.get_sofa_score(val, self.args.threshold_missing)
            sofa_all = np.insert(sub_score, len(sub_score), sofa)
            sofa_all_time.append(sofa_all)
        # print(len(sofa_all_time))
        sofa_all_time = pd.DataFrame(np.array(sofa_all_time), columns=columns_sofa)
        return sofa_all_time

    def __str__(self):
        return 'The class to calculate clinical score for sepsis. such as SOFA, qSOFA, mLODS, SIRS, NEWS.'


class PreDataset:

    """
        This class is used to pre-process the sepsis related data from different data. Such as yfy, eICU, MIMIC-III,
        MIMIC-IV. Here is the parameters:

        args.read_path: the individual patient source path, format -> 'csv'.
        args.method_merge: 'first/last/mean/max/min', how to merge the data.
        args.merge_time_window: set the time interval to merge the data, format: H -> hour.
        args.sample_time_window: set tge time interval to sample the data, format: H -> hour.
        args.predict_time_window
    """

    feature_static = ['subjectid', 'age', 'gender', 'height', 'weight', 'bmi', 'is_sepsis', 'center']  # 8
    feature_vital = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'gcs', 'vent', 'dobutamine',
                     'dopamine', 'epinephrine', 'norepinephrine', 'outputtotal']  # 13
    feature_lab = ['pco2', 'po2', 'spo2', 'fio2', 'totalco2', 'ph', 'lactate', 'baseexcess', 'ab', 'wbc',
                   'neutrophils', 'lymphocytes', 'monocytes', 'basophils', 'eosinophils',
                   'fibrinogen', 'pt', 'ptt', 'inr', 'platelet', 'rbc', 'hematocrit',
                   'hemoglobin', 'mch', 'mchc', 'mcv', 'rdw', 'alt', 'ast', 'alp',
                   'amylase', 'bilirubin_total', 'bilirubin_direct', 'ck_cpk', 'ck_mb',
                   'troponin_t', 'albumin', 'total_protein', 'cholesterol', 'aniongap',
                   'bicarbonate', 'bun', 'creatinine', 'glucose', 'calcium', 'chloride',
                   'sodium', 'potassium', 'magnesium', 'phosphate']  # 50
    feature_dynamic = feature_vital + feature_lab
    feature_left = ['charttime']  # 2
    feature_sofa = ['po2_fio2_ratio_last', 'vent_last', 'platelet_min', 'bilirubin_total_max', 'mbp_min',
                    'dopamine_max', 'dobutamine_last', 'epinephrine_max', 'norepinephrine_max', 'gcs_min',
                    'creatinine_max', 'outputtotal_sum']
    feature_sofa_score = ['respiration', 'coagulation', 'liver', 'cardiovascular', 'neurological', 'renal', 'sofa']
    feature_all = ['']

    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(args.read_path)
        self.sepsis_merge = self.data_merge_annotation()
        self.sepsis_internal_center, self.sepsis_external_center = self.data_center_select()
        self.sepsis_dataset_internal = self.get_labeled_data(self.sepsis_internal_center, args.internal_center)
        self.sepsis_dataset_external = self.get_labeled_data(self.sepsis_external_center, args.external_center)

    def __len__(self):
        return len(self.sepsis_merge)

    def __getitem__(self, index):
        return self.sepsis_merge.values[index]

    def __str__(self):
        return f'This is the class of pre-processing Sepsis Dataset for different data center.'

    def __repr__(self):
        result = 'The class of pre processing sepsis data'

    def sofa_individual_sample(self, data_individual):
        data_individual_dynamic_sofa = self.data_agg_reshape(data_individual[self.feature_dynamic].resample(
            f'{self.args.merge_time_window}H').agg(self.args.method_merge))
        data_individual_dynamic_sofa['po2_fio2_ratio_last'] = \
            data_individual_dynamic_sofa['po2_last'] / data_individual_dynamic_sofa['fio2_last']
        data_individual_dynamic_sofa['outputtotal_sum'] = \
            data_individual['outputtotal'].resample(f'{self.args.merge_time_window}H').sum().apply(
            lambda x: pd.NA if x == 0 else x) * 24 / self.args.merge_time_window
        data_individual_dynamic_sofa = data_individual_dynamic_sofa[self.feature_sofa]
        data_individual_dynamic_sofa.columns = [col + '_' + 'merge' for col in data_individual_dynamic_sofa.columns]
        data_individual_dynamic_sofa_grade = ClinicalScore(self.args).sofa_score(data_individual_dynamic_sofa.values)
        data_individual_dynamic_sofa.index = \
            data_individual_dynamic_sofa.index + \
            pd.to_timedelta(f'{self.args.merge_time_window - self.args.sample_time_window}H')
        data_individual_dynamic_sofa_grade.index = data_individual_dynamic_sofa.index
        # data_individual_dynamic_sofa_grade = \
        #     data_individual_dynamic_sofa.resample(f'{self.args.sample_time_window}H').asfreq()
        data_individual_dynamic_sofa_all = pd.concat((data_individual_dynamic_sofa, data_individual_dynamic_sofa_grade),
                                                     axis=1)
        return data_individual_dynamic_sofa_all

    @staticmethod
    def data_agg_reshape(data_col):
        col_new = []
        for col in data_col.columns:
            col_new.append('_'.join(col))
        data_col.columns = col_new
        return data_col

    @staticmethod
    def sepsis_sofa_mark(sofa):
        if pd.isna(sofa):
            sepsis = pd.NA
        elif sofa >= 2:
            sepsis = 1
        else:
            sepsis = 0
        return sepsis

    def data_merge_annotation(self, write_path='./data/merged/'):
        print('starting to merge sample for different medical data center:')
        name = f'sepsis_merged_{self.args.merge_time_window}_{self.args.sample_time_window}_' \
               f'{self.args.threshold_missing}.csv'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        if not os.path.exists(write_path + name):
            data_individual_merge_all = []
            subjectiid = pd.unique(self.data['subjectid'])
            self.data['charttime'] = pd.to_timedelta(self.data['charttime'], unit='H')
            for iid in tqdm(subjectiid):
                data_individual = self.data.iloc[np.where(self.data['subjectid'] == iid)[0], :]
                data_individual.set_index('charttime', inplace=True)
                data_individual_static = data_individual[self.feature_static]
                data_individual_dynamic = self.data_agg_reshape(data_individual[self.feature_dynamic]
                                                                .resample(f'{self.args.sample_time_window}H')
                                                                .agg(self.args.method_merge))
                # data_individual_merge = pd.concat(())
                data_individual_dynamic_sofa_grade = self.sofa_individual_sample(data_individual)
                if data_individual_dynamic.index[-1] < data_individual_dynamic_sofa_grade.index[-1]:
                    index_add = pd.timedelta_range(start=data_individual_dynamic.index[-1],
                                                   end=data_individual_dynamic_sofa_grade.index[-1],
                                                   freq=f'{self.args.sample_time_window}H', closed='right')
                    # print(index_add)
                    for ind in index_add:
                        data_individual_dynamic.loc[ind, :] = pd.NA

                data_individual_dynamic.loc[data_individual_dynamic_sofa_grade.index,
                                            data_individual_dynamic_sofa_grade.columns] = \
                    data_individual_dynamic_sofa_grade

                data_individual_static_select = pd.DataFrame(np.repeat(data_individual_static.values[[0]],
                                                                       data_individual_dynamic.shape[0], axis=0),
                                                             columns=data_individual_static.columns)
                data_individual_static_select.index = data_individual_dynamic.index
                data_individual_merge = pd.concat((data_individual_static_select, data_individual_dynamic), axis=1)
                # data_individual_merge_all = pd.concat((data_individual_merge_all, data_individual_merge), axis=0)
                data_individual_merge_all.append(data_individual_merge)
            data_individual_merge_all = pd.concat(data_individual_merge_all, axis=0)
            data_individual_merge_all = data_individual_merge_all.reset_index()
            data_individual_merge_all['sepsis'] = data_individual_merge_all['sofa'].apply(self.sepsis_sofa_mark)
            data_individual_merge_all['charttime'] = \
                data_individual_merge_all['charttime'].apply(lambda x: x.days * 24 + x.seconds / 3600)
            data_individual_merge_all.to_csv(write_path + name, index=False)
        else:
            data_individual_merge_all = pd.read_csv(write_path + name)
        print('down!')
        return data_individual_merge_all

    @staticmethod
    def time_delta_float(time_list):
        index_float = []
        for val in time_list:
            index_float.append(val.day * 24 + val.seconds / 24)
        return index_float

    def data_center_select(self, write_path='./data/center/'):
        print('starting to select the data center:')
        # center_dict = {'eicu': 0, 'xjtu': 1, 'mimic3cv': 2, 'mimiciv': 3}
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        str_internal = '_'.join(self.args.internal_center)
        str_external = '_'.join(self.args.external_center)
        name_internal = f'sepsis_merged_{self.args.merge_time_window}_{self.args.sample_time_window}_' \
                        f'{self.args.threshold_missing}_{str_internal}.csv'
        name_external = f'sepsis_merged_{self.args.merge_time_window}_{self.args.sample_time_window}_' \
                        f'{self.args.threshold_missing}_{str_external}.csv'
        if not os.path.exists(write_path + name_internal):
            data_internal_center = []
            for center in self.args.internal_center:
                data_internal_center.append(self.sepsis_merge.loc[self.sepsis_merge['center'] == center, :])
            data_internal_center = pd.concat(data_internal_center)
            data_internal_center.to_csv(write_path + name_internal, index=False)
        else:
            data_internal_center = pd.read_csv(write_path + name_internal)
        if not os.path.exists(write_path + name_external):
            data_external_center = []
            for center in self.args.external_center:
                data_external_center.append(self.sepsis_merge.loc[self.sepsis_merge['center'] == center, :])
            data_external_center = pd.concat(data_external_center)
            data_external_center.to_csv(write_path + name_external, index=False)
        else:
            data_external_center = pd.read_csv(write_path + name_external)
        print('down!')
        return data_internal_center, data_external_center

    def get_labeled_data(self, data, center, write_path='./data/processed/'):
        print('starting to get labeled data')
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        subjectiid = pd.unique(data['subjectid'])
        columns_selected = data.columns.to_list()
        columns_selected.remove('subjectid')
        columns_selected.remove('center')
        columns_selected.remove('charttime')
        patients = []
        for iid in subjectiid:
            data_individual = data.iloc[np.where(data['subjectid'] == iid)[0], :]
            data_individual = data_individual.reset_index(drop=True)
            time_step = int(self.args.merge_time_window / self.args.sample_time_window)
            time_adopt_step = int(self.args.adopt_time_window / self.args.sample_time_window)
            for j in range(time_step + time_adopt_step,
                           data_individual.shape[0] + time_step, time_step):
                if not pd.isna(data_individual.loc[j - 1, 'sepsis']):
                    record_id = iid
                    tt = torch.tensor(data_individual.loc[j - time_step - time_adopt_step: j - time_step - 1,
                                      'charttime'].values, dtype=torch.float32)
                    vals = data_individual.loc[j - time_step - time_adopt_step: j - time_step - 1, columns_selected]
                    masks = np.ones(vals.shape)
                    masks[np.where(pd.isna(vals))] = 0
                    vals = vals.fillna(0).values
                    vals = torch.tensor(vals, dtype=torch.float32)
                    masks = torch.tensor(masks, dtype=torch.float32)
                    labels = torch.tensor(data_individual.loc[j - 1, 'sepsis'], dtype=torch.float32)
                    sofas = torch.tensor(data_individual.loc[j - 1, 'sofa'], dtype=torch.float32)
                    patients.append((record_id, tt, vals, masks, labels, sofas))
        name_center = '_'.join(center)
        center_path = f'sepsis_merged_{self.args.merge_time_window}_{self.args.sample_time_window}_' \
                      f'{self.args.threshold_missing}_{name_center}.pt'
        torch.save(patients, write_path + center_path)
        print(f'the number of center "{name_center}": {len(patients)}')
        print('down!')
        return patients


if __name__ == '__main__':
    clin_score = SOFA()
