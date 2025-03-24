import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.datasets import dump_svmlight_file

def load_csv(file_path, column_names):
    return pd.read_csv(file_path, encoding='ISO-8859-1', sep=';', dtype={'User-ID': str}, names=column_names, header=0)

def process_users():
    all_users = load_csv("../../datasets/Users.csv", ['User-ID', 'Age'])
    users_with_ages = all_users[all_users['Age'].notna()]
    users_no_ages = all_users[all_users['Age'].isna()]
    
    users_with_ages.to_csv('users_with_age_data.csv', index=False)
    users_no_ages.to_csv('users_without_age_data.csv', index=False)
    
    print(f"Users with ages: {len(users_with_ages)}, Users without ages: {len(users_no_ages)}")
    
    return all_users, users_with_ages, users_no_ages

def process_ratings(aged_users, missing_age_users):
    all_ratings = load_csv("../../datasets/Ratings.csv", ['User-ID', 'ISBN', 'Book-Rating'])
    
    ratings_with_ages = all_ratings[all_ratings['User-ID'].isin(aged_users['User-ID'])]
    ratings_no_ages = all_ratings[all_ratings['User-ID'].isin(missing_age_users['User-ID'])]
    
    ratings_with_ages.to_csv('ratings_with_age_data.csv', index=False)
    ratings_no_ages.to_csv('ratings_without_age_data.csv', index=False)
    
    print(f"Ratings for users with ages: {len(ratings_with_ages)}, Ratings for users without ages: {len(ratings_no_ages)}")
    
    return all_ratings

def prepare_combined_data(rating_data, user_data):
    user_idx = {uid: i for i, uid in enumerate(rating_data['User-ID'].unique())}
    book_idx = {isbn: i for i, isbn in enumerate(rating_data['ISBN'].unique())}
    
    rating_data['user_index'] = rating_data['User-ID'].map(user_idx)
    rating_data['book_index'] = rating_data['ISBN'].map(book_idx)
    
    combined_data = pd.merge(rating_data, user_data[['User-ID', 'Age']], on='User-ID', how='left')
    combined_data['Age'] = pd.to_numeric(combined_data['Age'].fillna(0), errors='coerce').fillna(0)
    
    return combined_data, user_idx, book_idx

def generate_libsvm_files(combined_data, user_indices, book_indices):
    rating_matrix = coo_matrix(
        (combined_data['Book-Rating'], (combined_data['user_index'], combined_data['book_index'])),
        shape=(len(user_indices), len(book_indices))
    )
    
    age_values = combined_data.drop_duplicates('user_index').set_index('user_index')['Age']
    age_list = age_values.reindex(range(rating_matrix.shape[0]), fill_value=0).tolist()
    
    dump_svmlight_file(rating_matrix, age_list, "user_book_ratings.libsvm", zero_based=True)
    print("LIBSVM file: user_book_ratings.libsvm")
    
    idx_with_ages = [i for i, age in enumerate(age_list) if age > 0]
    idx_no_ages = [i for i, age in enumerate(age_list) if age == 0]
    
    matrix_with_ages = rating_matrix.tocsr()[idx_with_ages]
    matrix_no_ages = rating_matrix.tocsr()[idx_no_ages]
    ages_with = [age_list[i] for i in idx_with_ages]
    ages_without = [age_list[i] for i in idx_no_ages]
    
    dump_svmlight_file(matrix_with_ages, ages_with, "ratings_with_ages.libsvm", zero_based=True)
    dump_svmlight_file(matrix_no_ages, ages_without, "ratings_no_ages.libsvm", zero_based=True)
    
    print("LIBSVM file with ages: ratings_with_ages.libsvm")
    print("LIBSVM file with missing ages: ratings_no_ages.libsvm")


all_users, aged_users, missing_age_users = process_users()
rating_data = process_ratings(aged_users, missing_age_users)
combined_data, user_indices, book_indices = prepare_combined_data(rating_data, all_users)
generate_libsvm_files(combined_data, user_indices, book_indices)