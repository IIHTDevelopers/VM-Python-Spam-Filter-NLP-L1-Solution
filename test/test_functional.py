import unittest
from test.TestUtils import TestUtils  # Yaksha Test Utility
from main import (
    load_dataset,
    preprocess_text,
    preprocess_dataset,
    split_dataset,
    extract_features,
    train_model,
    evaluate_model,
    get_total_messages,
    get_spam_count,
    get_ham_count,
    get_spam_ratio,
    get_ham_ratio,
)

class TestSpamClassifier(unittest.TestCase):

    def test_load_dataset(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()
            expected_rows = 10
            if len(df) == expected_rows:
                test_obj.yakshaAssert("TestLoadDataset", True, "functional")
                print("TestLoadDataset = Passed")
            else:
                test_obj.yakshaAssert("TestLoadDataset", False, "functional")
                print("TestLoadDataset = Failed")
        except Exception as e:
            print(f"Error in test_load_dataset: {e}")
            test_obj.yakshaAssert("TestLoadDataset", False, "functional")
            print("TestLoadDataset = Failed")

    def test_preprocess_text(self):
        test_obj = TestUtils()
        try:
            text = "Free entry in a contest to win $1000!"
            processed_text = preprocess_text(text)
            expected_text = "free entry in a contest to win "
            if processed_text == expected_text:
                test_obj.yakshaAssert("TestPreprocessText", True, "functional")
                print("TestPreprocessText = Passed")
            else:
                test_obj.yakshaAssert("TestPreprocessText", False, "functional")
                print("TestPreprocessText = Failed")
        except Exception as e:
            print(f"Error in test_preprocess_text: {e}")
            test_obj.yakshaAssert("TestPreprocessText", False, "functional")
            print("TestPreprocessText = Failed")

    def test_preprocess_dataset(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()
            df = preprocess_dataset(df)
            if "Processed_Text" in df.columns:
                test_obj.yakshaAssert("TestPreprocessDataset", True, "functional")
                print("TestPreprocessDataset = Passed")
            else:
                test_obj.yakshaAssert("TestPreprocessDataset", False, "functional")
                print("TestPreprocessDataset = Failed")
        except Exception as e:
            print(f"Error in test_preprocess_dataset: {e}")
            test_obj.yakshaAssert("TestPreprocessDataset", False, "functional")
            print("TestPreprocessDataset = Failed")

    def test_split_dataset(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()
            df = preprocess_dataset(df)
            X_train, X_test, y_train, y_test = split_dataset(df)
            if len(X_train) > len(X_test):
                test_obj.yakshaAssert("TestSplitDataset", True, "functional")
                print("TestSplitDataset = Passed")
            else:
                test_obj.yakshaAssert("TestSplitDataset", False, "functional")
                print("TestSplitDataset = Failed")
        except Exception as e:
            print(f"Error in test_split_dataset: {e}")
            test_obj.yakshaAssert("TestSplitDataset", False, "functional")
            print("TestSplitDataset = Failed")

    def test_extract_features(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()
            df = preprocess_dataset(df)
            X_train, X_test, y_train, y_test = split_dataset(df)
            X_train_vectors, X_test_vectors, vectorizer = extract_features(X_train, X_test)

            # Check if vectors are not empty and are proper sparse matrices
            if hasattr(X_train_vectors, 'shape') and hasattr(X_test_vectors, 'shape'):
                if X_train_vectors.shape[0] == len(X_train) and X_test_vectors.shape[0] == len(X_test):
                    test_obj.yakshaAssert("TestExtractFeatures", True, "functional")
                    print("TestExtractFeatures = Passed")
                else:
                    test_obj.yakshaAssert("TestExtractFeatures", False, "functional")
                    print("TestExtractFeatures = Failed")
            else:
                print("Error: Feature extraction did not return sparse matrices.")
                test_obj.yakshaAssert("TestExtractFeatures", False, "functional")
                print("TestExtractFeatures = Failed")
        except Exception as e:
            print(f"Error in test_extract_features: {e}")
            test_obj.yakshaAssert("TestExtractFeatures", False, "functional")
            print("TestExtractFeatures = Failed")

    def test_train_model(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()
            df = preprocess_dataset(df)
            X_train, X_test, y_train, y_test = split_dataset(df)
            X_train_vectors, X_test_vectors, vectorizer = extract_features(X_train, X_test)
            model = train_model(X_train_vectors, y_train)
            if model is not None:
                y_pred = model.predict(X_train_vectors)
                if len(y_pred) == len(y_train):
                    test_obj.yakshaAssert("TestTrainModel", True, "functional")
                    print("TestTrainModel = Passed")
                else:
                    test_obj.yakshaAssert("TestTrainModel", False, "functional")
                    print("TestTrainModel = Failed")
            else:
                print("Error: Model is None after training.")
                test_obj.yakshaAssert("TestTrainModel", False, "functional")
                print("TestTrainModel = Failed")
        except Exception as e:
            print(f"Error in test_train_model: {e}")
            test_obj.yakshaAssert("TestTrainModel", False, "functional")
            print("TestTrainModel = Failed")

    def test_evaluate_model(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()
            df = preprocess_dataset(df)
            X_train, X_test, y_train, y_test = split_dataset(df)
            X_train_vectors, X_test_vectors, vectorizer = extract_features(X_train, X_test)
            model = train_model(X_train_vectors, y_train)
            accuracy, correct_predictions, incorrect_predictions = evaluate_model(model, X_test_vectors, y_test)
            expected_accuracy = 0.67
            if round(accuracy, 2) == expected_accuracy:
                test_obj.yakshaAssert("TestEvaluateModel", True, "functional")
                print("TestEvaluateModel = Passed")
            else:
                test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
                print("TestEvaluateModel = Failed")
        except Exception as e:
            print(f"Error in test_evaluate_model: {e}")
            test_obj.yakshaAssert("TestEvaluateModel", False, "functional")
            print("TestEvaluateModel = Failed")

    def test_analytical_functions(self):
        test_obj = TestUtils()
        try:
            df = load_dataset()

            # Total messages
            total_messages = get_total_messages(df)
            if total_messages == 10:
                test_obj.yakshaAssert("TestTotalMessages", True, "functional")
                print("TestTotalMessages = Passed")
            else:
                test_obj.yakshaAssert("TestTotalMessages", False, "functional")
                print("TestTotalMessages = Failed")

            # Spam count
            spam_count = get_spam_count(df)
            if spam_count == 6:
                test_obj.yakshaAssert("TestSpamCount", True, "functional")
                print("TestSpamCount = Passed")
            else:
                test_obj.yakshaAssert("TestSpamCount", False, "functional")
                print("TestSpamCount = Failed")

            # Ham count
            ham_count = get_ham_count(df)
            if ham_count == 4:
                test_obj.yakshaAssert("TestHamCount", True, "functional")
                print("TestHamCount = Passed")
            else:
                test_obj.yakshaAssert("TestHamCount", False, "functional")
                print("TestHamCount = Failed")

            # Spam ratio
            spam_ratio = get_spam_ratio(df)
            if round(spam_ratio, 2) == 60.00:
                test_obj.yakshaAssert("TestSpamRatio", True, "functional")
                print("TestSpamRatio = Passed")
            else:
                test_obj.yakshaAssert("TestSpamRatio", False, "functional")
                print("TestSpamRatio = Failed")

            # Ham ratio
            ham_ratio = get_ham_ratio(df)
            if round(ham_ratio, 2) == 40.00:
                test_obj.yakshaAssert("TestHamRatio", True, "functional")
                print("TestHamRatio = Passed")
            else:
                test_obj.yakshaAssert("TestHamRatio", False, "functional")
                print("TestHamRatio = Failed")

        except Exception as e:
            print(f"Error in test_analytical_functions: {e}")
            test_obj.yakshaAssert("TestAnalyticalFunctions", False, "functional")
            print("TestAnalyticalFunctions = Failed")


if __name__ == "__main__":
    unittest.main()
