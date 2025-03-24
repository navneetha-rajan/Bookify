# Bookify
# Book Crossing Data Mining & Machine Learning Project

This repository contains a complete data processing pipeline for the BookCrossing dataset. The primary goal of the project is to leverage data mining and machine learning techniques to analyze a large dataset of book ratings and enable downstream tasks such as building personalized recommender systems, discovering user segments, or predicting user age based on reading patterns.

---


## Project Overview

This project involves analyzing a large dataset of book ratings using data mining and machine learning techniques. The Book Crossing dataset, sourced from Kaggle, consists of detailed records for over 270,000 books, 270,000+ users, and more than 1 million ratings on a scale from 0 to 10. The repository demonstrates robust data preparation methods to facilitate various analytical tasks.

---

## Project Goals

Based on the project description, this dataset can serve multiple purposes. The goals include:
- **Building a Personalized Recommender System:**  
  Develop a system that recommends books to users based on their individual tastes and historical ratings.
- **User Profiling through Segmentation:**  
  Discover different user groups by analyzing reading history and ratings to create detailed user profiles.
- **Age Prediction via Regression Analysis:**  
  Build a regression model to estimate a user’s age based on their rating behavior.

In this project, teams may choose one of these tasks and further perform data analysis and evaluation based on the prepared dataset.

---

## Dataset Description

The [Book Crossing dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset?resource=download) from Kaggle includes three primary files:

- **Books.csv:**  
  Contains metadata for over 270,000 books (title, description, author, ISBN, etc.).
- **Users.csv:**  
  Lists user IDs along with age information.
- **Ratings.csv:**  
  Provides more than 1 million user ratings for various books on a scale from 0 to 10.

---

## Features & Functionality

- **Data Splitting & Cleaning:**  
  - Reads the Users and Ratings datasets with consistent data types and encoding.
  - Separates users into two groups: those with age data and those without.
- **Data Transformation:**  
  - Maps `User-ID` and `ISBN` to numerical indices.
  - Merges user age information with ratings.
  - Converts the ratings into a sparse matrix format.
- **Data Export:**  
  - Exports the full processed data as a LIBSVM file.
  - Generates separate LIBSVM files for users with existing age information and users with missing age data.
  
These steps facilitate applying various machine learning tasks like recommendation systems, clustering for user profiling, or regression analysis for age prediction.

