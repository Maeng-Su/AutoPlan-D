# ga_util.py

# TODO: 파일 상단에 이 파일에서 사용하는 모든 표준 라이브러리 및 서드파티 라이브러리를 import 해주세요.
# 예: import pandas as pd, import numpy as np, import datetime
# GeneticAlgorithm 클래스를 사용한다면 from .ga_core import GeneticAlgorithm (또는 from ga_core import GeneticAlgorithm)
# generation_xijt, genetic_algorithm 함수는 GeneticAlgorithm 클래스의 메소드가 될 예정이므로,
# find_hyper_parameters 함수는 GeneticAlgorithm 클래스 인스턴스를 통해 이 기능들을 호출해야 합니다.
# 따라서 find_hyper_parameters에서 직접 이 함수들을 호출하는 부분은 수정이 필요합니다.

import pandas as pd # set_hyper_parameters에서 사용
import numpy as np  # set_hyper_parameters에서 사용
import datetime     # find_hyper_parameters에서 사용
from ga_core import GeneticAlgorithm # find_hyper_parameters에서 GA 클래스를 사용하기 위해 필요

def preprocess_ga_inputs_dense(dataset):
    """
    GA 입력 데이터를 전처리합니다 (Dense 방식).
    cit, pit, dit, mijt 딕셔너리는 I, T, J의 모든 조합에 대해 값을 가지며,
    원본 데이터에 해당 조합이 없으면 0으로 채웁니다.
    """
    I_unique = sorted(list(set(dataset['item'])))
    T_unique = sorted(list(set(dataset['time'])))
    J_unique = sorted(list(set(dataset['machine'])))

    idx_item_time = pd.MultiIndex.from_product([I_unique, T_unique], names=['item', 'time'])

    cit_series = dataset.groupby(['item', 'time'])['cost'].first()
    cit = cit_series.reindex(idx_item_time, fill_value=0).to_dict()

    pit_series = dataset.groupby(['item', 'time'])['urgent'].first()
    pit = pit_series.reindex(idx_item_time, fill_value=0).to_dict()

    # dit는 데이터 특성에 따라 .first() 또는 .sum() 사용
    dit_series = dataset.groupby(['item', 'time'])['qty'].first() 
    dit = dit_series.reindex(idx_item_time, fill_value=0).to_dict()

    # mijt 계산 (Dense 방식)
    idx_item_machine = pd.MultiIndex.from_product([I_unique, J_unique], names=['item', 'machine'])
    capacity_series_from_data = dataset.groupby(['item', 'machine'])['capacity'].first()
    # (item,machine) 조합에 대해 capacity가 0이거나 없으면 0으로 채움
    capacity_map_dense = capacity_series_from_data.reindex(idx_item_machine, fill_value=0) 
    
    mijt = {}
    for (i_item, j_machine), capacity_value in capacity_map_dense.items():
        # capacity_value가 0.0이면 정수 0으로, 아니면 원래 float 값 사용
        value_to_assign = 0 if capacity_value == 0.0 else capacity_value
        for t_time in T_unique:
            mijt[(i_item, j_machine, t_time)] = value_to_assign

    return T_unique, I_unique, J_unique, cit, pit, dit, mijt
    
def preprocess_ga_inputs_sparse(dataset):
    """
    GA 입력 데이터를 전처리합니다 (Sparse 방식).
    cit, pit, dit, mijt 딕셔너리는 원본 데이터에 실제로 존재하는 조합에 대해서만 값을 가집니다.
    GA 로직에서는 .get(key, 0)을 사용하여 없는 키에 접근해야 합니다.
    """
    I_unique = sorted(list(set(dataset['item'])))
    T_unique = sorted(list(set(dataset['time'])))
    J_unique = sorted(list(set(dataset['machine']))) # mijt 생성 시 필요할 수 있음

    # (item, time)으로 그룹핑하고 각 그룹의 첫 번째 값을 가져와 바로 딕셔너리로 변환
    cit = dataset.groupby(['item', 'time'])['cost'].first().to_dict()
    pit = dataset.groupby(['item', 'time'])['urgent'].first().to_dict()
    # dit는 데이터 특성에 따라 .first() 또는 .sum() 사용
    dit = dataset.groupby(['item', 'time'])['qty'].first().to_dict() 

    # mijt 계산 (Sparse 방식)
    # (item, machine) 조합의 capacity를 가져오되, 데이터에 있는 조합만 처리
    capacity_map_sparse = dataset.groupby(['item', 'machine'])['capacity'].first().to_dict()
    
    mijt = {}
    for (i_item, j_machine), capacity_value in capacity_map_sparse.items():
        # 여기서 capacity_value가 0인 것을 포함할지 여부는 네 결정에 따름.
        # 네가 "존재 하는 (아이템, 머신) 중에 실제로 capacity_value가 0인 것이 있을 수 있잖아. 그럼 이건 포함 해야 하지 않니"
        # 라고 했으니, capacity_value > 0 같은 필터 없이 그대로 사용.
        for t_time in T_unique: # T_unique는 전체 시간 범위를 사용
            mijt[(i_item, j_machine, t_time)] = capacity_value
            
    return T_unique, I_unique, J_unique, cit, pit, dit, mijt

def dict_to_list(data_dict, keys_list, default_value=0):
    """딕셔너리를 제공된 키 리스트의 순서에 따라 리스트로 변환 합니다.

    """
    return [data_dict.get(key, default_value) for key in keys_list]

def list_to_dict(data_list, keys_list):
    """리스트를 제공된 키 리스트에 매핑하여  딕셔너리로 변환 합니다.

    """
    if len(data_list) != len(keys_list):
        min_len = min(len(data_list), len(keys_list))
        print(f"Warning: list_to_dict에서 리스트와 키 리스트의 길이가 다릅니다. 짧은 쪽({min_len}개)에 맞춰 처리합니다.")
        return {keys_list[i]: data_list[i] for i in range(min_len)}

    return {key: data_list[i] for i, key in enumerate(keys_list)}

def set_hyper_parameters():
    # 하이퍼파라미터 설정
    hyper_parameters = pd.DataFrame({
        'index':    ['index_1', 'index_2', 'index_3', 'index_4',
                     'index_5', 'index_6', 'index_7', 'index_8'],
        'n_iter':   [500, 500, 500, 500, 500, 500, 500, 500],
        'n_pop':    [10, 20, 40, 20, 20, 20, 20, 20],
        'r_cross':  [0.4, 0.4, 0.4, 0.1, 0.2, 0.3, 0.4, 0.4],
        'r_mut':    [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.6]
    })

    # 결과 기록용 열 추가
    hyper_parameters['objective'] = np.nan # numpy 사용 확인 (import numpy as np 필요)
    hyper_parameters['time'] = np.nan    # numpy 사용 확인

    return hyper_parameters

# TODO: find_hyper_parameters 함수는 GeneticAlgorithm 클래스를 사용하도록 대대적인 수정이 필요합니다.
#       현재는 전역 변수 및 이전 방식의 함수 호출에 의존하고 있습니다.
def find_hyper_parameters(dataset_for_ga_sample, hyper_parameters_df): # 인자 이름 변경 및 dataset_for_ga_sample 추가
    # TODO: from ga_core import GeneticAlgorithm (또는 from .ga_core import ...) 임포트 필요.
    #       이 파일 상단이나 이 함수 바로 위에.
    from ga_core import GeneticAlgorithm # 예시 위치, 실제로는 파일 상단이 좋음

    log_list = []

    # TODO: 데이터 전처리는 하이퍼파라미터 루프 *전에* 한 번만 수행하는 것이 효율적입니다.
    #       (모든 하이퍼파라미터 조합이 동일한 데이터셋에 대해 테스트된다고 가정)
    #       T, I, J, cit, pit, dit, mijt = preprocess_ga_inputs(dataset_for_ga_sample)
    #       이 값들을 아래 루프에서 GeneticAlgorithm 인스턴스 생성 시 사용합니다.
    #       (아래 코드에서는 이 부분이 누락되어 있어 추가 필요)
    
    # 임시로 데이터 전처리 호출 (실제로는 루프 전에 한 번만!) - 이 부분은 구조를 다시 잡아야 함
    # 아래 루프 안에서 GA 인스턴스를 만들 때 필요한 T,I,J,cit 등을 어떻게 가져올지 명확히 해야 함.
    # 여기서는 T_data, I_data ... 등이 이미 준비되었다고 가정하고 진행. (실제로는 preprocess_ga_inputs 호출 필요)
    
    # ---- 이 아래는 수정된 로직 예시 ----
    # 1. 데이터 샘플에 대한 전처리 (루프 전에 한 번)
    T_data, I_data, J_data, cit_data, pit_data, dit_data, mijt_data = preprocess_ga_inputs(dataset_for_ga_sample)
    # xijt_keys, mijt_keys 등도 GeneticAlgorithm 클래스 생성자에서 내부적으로 처리하거나,
    # 혹은 여기서 생성해서 전달할 수 있습니다. 여기서는 클래스 내부에서 처리한다고 가정.

    for i in range(len(hyper_parameters_df)): # 인자로 받은 hyper_parameters_df 사용
        parameter = hyper_parameters_df.iloc[i]
        index_nm = parameter['index']
        print(f'{index_nm}')

        start = datetime.datetime.now()

        # TODO: 전역 변수(xijt_keys, mijt_keys) 및 전역 함수(generation_xijt, genetic_algorithm, mijt) 사용 제거.
        #       GeneticAlgorithm 클래스의 인스턴스를 생성하고 solve() 메소드를 호출해야 합니다.
        
        # 하이퍼파라미터 설정 (DataFrame에서 가져올 때 타입 변환 주의, 예: int로)
        n_iter_val = int(parameter['n_iter'])
        n_pop_val = int(parameter['n_pop'])
        r_cross_val = parameter['r_cross']
        r_mut_val = parameter['r_mut']

        # GeneticAlgorithm 인스턴스 생성
        ga_instance = GeneticAlgorithm(
            mijt_data=mijt_data, 
            I_set=I_data, T_set=T_data, J_set=J_data,
            cit_data=cit_data, pit_data=pit_data, dit_data=dit_data,
            n_iter=n_iter_val, n_pop=n_pop_val, 
            r_cross=r_cross_val, r_mut=r_mut_val
            # xijt_keys, mijt_keys는 GeneticAlgorithm 클래스의 __init__에서
            # 내부적으로 생성하거나 None으로 두고 필요시 생성하도록 수정 가능
        )
        
        # 유전 알고리즘 실행
        best, score, log, log_detail = ga_instance.solve() # 클래스의 solve 메소드 호출

        # 실행 시간 및 결과 기록
        elapsed_time = datetime.datetime.now() - start
        hyper_parameters_df.loc[i, 'time'] = elapsed_time.total_seconds() # .total_seconds()로 숫자 변환
        hyper_parameters_df.loc[i, 'objective'] = score

        log_list.append(log)

    # 결과 출력
    return hyper_parameters_df, log_list