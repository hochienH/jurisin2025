#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <functional> // 支援 hash 函數
// 移除 <variant> 引入

// 定義 Citation 結構
struct Citation {
    std::string code;    // 法典
    std::string date;    // 日期
    std::string article; // 條號，改為 string

    bool operator==(const Citation& other) const {
        return code == other.code && article == other.article;
    }
    
    // 添加小於運算符以支援排序
    bool operator<(const Citation& other) const {
        if (code != other.code) {
            return code < other.code;
        }
        return article < other.article;
    }
};

// 為 Citation 新增 hash 函數
namespace std {
    template <>
    struct hash<Citation> {
        size_t operator()(const Citation& c) const {
            // 組合 code 和 article 的 hash
            size_t h1 = std::hash<std::string>()(c.code);
            size_t h2 = std::hash<std::string>()(c.article);
            return h1 ^ (h2 << 1);
        }
    };
}

// 定義 DataPoint 結構
struct DataPoint {
    std::string fileName;
    std::vector<double> textEmbeddings;
    std::vector<Citation> citations;  // 新增: 儲存引用的法條
};

// 計算歐幾里得距離
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// 計算 Acuracy - 移除 C++11 特性
double accuracy(const std::vector<Citation>& real, const std::vector<Citation>& predict) {
    double intersectCount = 0;
    for (size_t i = 0; i < real.size(); ++i) {
        for (size_t j = 0; j < predict.size(); ++j) {
            if (real[i] == predict[j]) {
                intersectCount++;
                break;
            }
        }
    }
    
    double unionCount = real.size() + predict.size() - intersectCount;
    
    if (unionCount == 0) return 0.0;
    return intersectCount / unionCount;
}

// 計算 Precision - 移除 C++11 特性
double precision(const std::vector<Citation>& real, const std::vector<Citation>& predict) {
    double intersectCount = 0;
    for (size_t i = 0; i < real.size(); ++i) {
        for (size_t j = 0; j < predict.size(); ++j) {
            if (real[i] == predict[j]) {
                intersectCount++;
                break;
            }
        }
    }
    
    if (predict.size() == 0) return 0.0;
    return intersectCount / predict.size();
}

// 計算 Recall - 移除 C++11 特性
double recall(const std::vector<Citation>& real, const std::vector<Citation>& predict) {
    double intersectCount = 0;
    for (size_t i = 0; i < real.size(); ++i) {
        for (size_t j = 0; j < predict.size(); ++j) {
            if (real[i] == predict[j]) {
                intersectCount++;
                break;
            }
        }
    }
    
    if (real.size() == 0) return 0.0;
    return intersectCount / real.size();
}

// 計算 Dice Coefficient - 移除 C++11 特性
double dice_coefficient(const std::vector<Citation>& real, const std::vector<Citation>& predict) {
    double intersectCount = 0;
    for (size_t i = 0; i < real.size(); ++i) {
        for (size_t j = 0; j < predict.size(); ++j) {
            if (real[i] == predict[j]) {
                intersectCount++;
                break;
            }
        }
    }
    
    if (real.size() + predict.size() == 0) return 0.0;
    return (2 * intersectCount) / (real.size() + predict.size());
}

// 解析引用法條字串 - 移除 C++11 特性
std::vector<Citation> parseCitations(const std::string& citationsStr) {
    std::vector<Citation> citations;
    std::string str = citationsStr;
    
    // 找到每一組括號的內容
    size_t pos = 0;
    while ((pos = str.find('(', pos)) != std::string::npos) {
        size_t end = str.find(')', pos);
        if (end == std::string::npos) break;
        
        // 取出括號內的內容
        std::string content = str.substr(pos + 1, end - pos - 1);
        
        // 處理單引號和逗號分隔的內容
        size_t start = 0;
        std::vector<std::string> parts;
        bool inQuotes = false;
        std::string current;
        
        // 逐字符處理內容
        for (size_t i = 0; i < content.size(); ++i) {
            char c = content[i];
            if (c == '\'') {
                inQuotes = !inQuotes;
                continue;
            }
            if (c == ',' && !inQuotes) {
                parts.push_back(current);
                current.clear();
                continue;
            }
            current += c;
        }
        if (!current.empty()) {
            parts.push_back(current);
        }
        
        // 去除每個部分的空白
        for (size_t i = 0; i < parts.size(); ++i) {
            std::string& part = parts[i];
            part.erase(0, part.find_first_not_of(" "));
            part.erase(part.find_last_not_of(" ") + 1);
        }
        
        // 確保我們有三個部分
        if (parts.size() >= 3) {
            Citation c;
            c.code = parts[0];
            c.date = parts[1];
            c.article = parts[2];
            citations.push_back(c);
        }
        
        pos = end + 1;
    }
    
    return citations;
}

// 讀取 CSV 檔案
std::vector<DataPoint> loadCSV(const std::string& filePath) {
    std::vector<DataPoint> dataPoints;
    std::ifstream file(filePath);
    std::string line;
    int lineNumber = 0;
    
    // 跳過標題行
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        lineNumber++;
        try {
            size_t firstQuote = line.find('\"');
            size_t secondQuote = line.find('\"', firstQuote + 1);
            size_t thirdQuote = line.find('\"', secondQuote + 1);
            size_t fourthQuote = line.find('\"', thirdQuote + 1);
            size_t fifthQuote = line.find('\"', fourthQuote + 1);
            size_t sixthQuote = line.find('\"', fifthQuote + 1);
            
            if (firstQuote == std::string::npos || secondQuote == std::string::npos ||
                thirdQuote == std::string::npos || fourthQuote == std::string::npos ||
                fifthQuote == std::string::npos || sixthQuote == std::string::npos) {
                std::cerr << "Invalid format at line " << lineNumber << std::endl;
                continue;
            }
            
            DataPoint dataPoint;
            // 提取檔名
            dataPoint.fileName = line.substr(firstQuote + 1, secondQuote - firstQuote - 1);
            
            // 提取 embedding 字串
            std::string textEmbeddingsStr = line.substr(thirdQuote + 1, fourthQuote - thirdQuote - 1);
            
            // 移除方括號
            textEmbeddingsStr.erase(std::remove(textEmbeddingsStr.begin(), textEmbeddingsStr.end(), '['), textEmbeddingsStr.end());
            textEmbeddingsStr.erase(std::remove(textEmbeddingsStr.begin(), textEmbeddingsStr.end(), ']'), textEmbeddingsStr.end());
            
            std::stringstream embeddingsStream(textEmbeddingsStr);
            std::string value;
            while (std::getline(embeddingsStream, value, ',')) {
                value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
                value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);
                if (!value.empty()) {
                    try {
                        dataPoint.textEmbeddings.push_back(std::stod(value));
                    } catch (const std::exception& e) {
                        std::cerr << "Error converting value '" << value << "' to double at line " 
                                << lineNumber << std::endl;
                    }
                }
            }
            
            // 提取引用法條字串
            std::string citationsStr = line.substr(fifthQuote + 1, sixthQuote - fifthQuote - 1);
            dataPoint.citations = parseCitations(citationsStr);
            
            if (!dataPoint.textEmbeddings.empty()) {
                dataPoints.push_back(dataPoint);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing line " << lineNumber << ": " << e.what() << std::endl;
        }
    }
    
    return dataPoints;
}

// 為鄰居排序定義一個比較器結構
struct NeighborComparer {
    bool operator()(const std::pair<int, double>& a, const std::pair<int, double>& b) const {
        return a.second < b.second; // 按距離由小到大排序
    }
};

// 將 target 的鄰居進行排序 - 移除 C++11 特性和 Lambda
std::vector<std::pair<int, double> > sortedNeighbors(
    const std::vector<DataPoint>& dataPoints,
    const DataPoint& target
) {
    std::vector<std::pair<int, double> > neighbors;
    
    // 計算目標點與所有其他點的距離
    for (size_t i = 0; i < dataPoints.size(); ++i) {
        if (dataPoints[i].fileName != target.fileName) {
            double dist = euclideanDistance(target.textEmbeddings, dataPoints[i].textEmbeddings);
            neighbors.push_back(std::make_pair(i, dist));
        }
    }
    
    // 根據距離由小到大排序，使用自定義比較器
    std::sort(neighbors.begin(), neighbors.end(), NeighborComparer());
    
    return neighbors;
}

// 找出距離最近的 k 個鄰居
std::vector<int> findKNearestNeighbors(
    std::vector<std::pair<int, double> > sortedNeighbors,
    int k
) {
    // 只保留前 k 個
    if (sortedNeighbors.size() > k) {
        sortedNeighbors.resize(k);
    }
    // 把鄰居的形狀變成只剩下 index
    std::vector<int> neighborIndices;
    for (size_t i = 0; i < sortedNeighbors.size(); ++i) {
        neighborIndices.push_back(sortedNeighbors[i].first);
    }
    return neighborIndices;
}

// 找出距離在 d 以內的所有鄰居
std::vector<int> findNeighborsWithinDistance(
    std::vector<std::pair<int, double> > sortedNeighbors,
    double d
) {
    int chunk_size = 0;
    for (size_t i = 0; i < sortedNeighbors.size(); ++i) {
        if (sortedNeighbors[i].second > d) {
            chunk_size = i;
            break;
        }
    }
    // 只保留距離在 d 以內的鄰居
    sortedNeighbors.resize(chunk_size);
    // 把鄰居的形狀變成只剩下 index
    std::vector<int> neighborIndices;
    for (size_t i = 0; i < sortedNeighbors.size(); ++i) {
        neighborIndices.push_back(sortedNeighbors[i].first);
    }
    return neighborIndices;
}

// 統計 neighbors 中出現的法條與頻次 - 移除 C++11 特性
std::unordered_map<Citation, int> countCitationsInNeighbors(
    const std::vector<DataPoint>& dataPoints,
    const std::vector<int>& neighbors
) {
    std::unordered_map<Citation, int> citationCount;
    
    for (size_t i = 0; i < neighbors.size(); ++i) {
        int neighborIndex = neighbors[i];
        for (size_t j = 0; j < dataPoints[neighborIndex].citations.size(); ++j) {
            const Citation& citation = dataPoints[neighborIndex].citations[j];
            citationCount[citation]++;
        }
    }
    
    return citationCount;
} 

// 自定義比較函數，用於排序 Citation-int 對
struct CitationFreqComparer {
    bool operator()(const std::pair<Citation, int>& a, const std::pair<Citation, int>& b) const {
        return a.second > b.second; // 從高到低排序頻次
    }
};

// 依據 citationCount 排序法條與頻次
std::vector<std::pair<Citation, int> > sortCitationsByFrequency(
    const std::unordered_map<Citation, int>& citationCount
) {
    std::vector<std::pair<Citation, int> > sortedCitations;
    
    for (std::unordered_map<Citation, int>::const_iterator it = citationCount.begin(); it != citationCount.end(); ++it) {
        sortedCitations.push_back(*it);
    }
    
    // 使用自定義比較函數而不是 std::greater
    std::sort(sortedCitations.begin(), sortedCitations.end(), CitationFreqComparer());
    
    return sortedCitations;
}

// 定義最佳化結構
struct OptimizationResult {
    int threshold;               // 最佳閾值（頻次）
    double coefficient;          // 最佳 Dice Coefficient
    std::vector<Citation> predictCitations;  // 最佳預測法條
};

// 從大到小，依序遍歷頻次 - 移除 C++11 特性
OptimizationResult predictCitationsByFrequency(
    const DataPoint& target,
    const std::vector<DataPoint>& dataPoints,
    const std::vector<std::pair<Citation, int> >& sortedCitations
) {
    // 定義回傳結構
    OptimizationResult result;
    // 將 target.citations 加入 vector 以便比較
    std::vector<Citation> targetCitations;
    for (size_t i = 0; i < target.citations.size(); ++i) {
        targetCitations.push_back(target.citations[i]);
    }
    // 定義 predictCitations 以儲存預測的法條
    std::vector<Citation> predictCitations;
    // 定義 bestPredictCitations 以儲存最佳 Dice Coefficient 的預測法條
    std::vector<Citation> bestPredictCitations;
    // 定義 maxDiceCoefficient 以儲存最佳 Dice Coefficient
    double maxDiceCoefficient = 0.0;
    // 定義 bestThreshold 以儲存最佳 Dice Coefficient 的預測頻次
    int bestThreshold = 0;
    
    // 由大到小頻次遍歷 sortedCitations
    for (size_t i = 0; i < sortedCitations.size(); ++i) {
        // 加入法條至預測法條
        predictCitations.push_back(sortedCitations[i].first);
        // 如果下一個元素的頻次不等於當前頻次，或是已經到達最後一個元素，則計算 Dice Coefficient
        if (i == sortedCitations.size() - 1 || sortedCitations[i].second != sortedCitations[i + 1].second) {
            double diceCoefficient = dice_coefficient(targetCitations, predictCitations);
            if (diceCoefficient > maxDiceCoefficient) {
                maxDiceCoefficient = diceCoefficient;
                bestThreshold = sortedCitations[i].second;
                bestPredictCitations = predictCitations;
            }
        }
    }
    // 回傳最佳化結果
    result.threshold = bestThreshold;
    result.coefficient = maxDiceCoefficient;
    result.predictCitations = bestPredictCitations;
    return result;    
}

// 定義實驗結果架構
struct ExperimentResult {
    union {
        int k;          // k-NN 的 k 值
        double d;       // 鄰居距離閾值
    };
    bool isK;           // 用來判斷儲存的是 k 還是 d
    double coefficient; // Dice Coefficient
    double precision;   // Precision
    double recall;      // Recall
    double accuracy;    // Accuracy
};

// 定義實驗一：運用固定鄰居數量預測法條集(k方法)
std::vector<ExperimentResult> experiment1(
    const DataPoint& target,
    const std::vector<DataPoint>& dataPoints,
    int maxK
) {
    // 定義回傳結果
    std::vector<ExperimentResult> results;
    // 定義 neighbors 以儲存鄰居
    std::vector<std::pair<int, double> > sorted_neighbors;
    std::vector<int> neighbors;
    // 排序鄰居
    sorted_neighbors = sortedNeighbors(dataPoints, target);
    // 依序處理每個 k 值
    for (int k = 1; k <= maxK; k++) {
        // 找出最近的 k 個鄰居
        neighbors = findKNearestNeighbors(sorted_neighbors, k);
        // 統計 neighbors 中出現的法條與頻次
        std::unordered_map<Citation, int> citationCount = countCitationsInNeighbors(dataPoints, neighbors);
        // 依據 citationCount 排序法條與頻次
        std::vector<std::pair<Citation, int> > sortedCitations = sortCitationsByFrequency(citationCount);
        // 預測法條集
        OptimizationResult optResult = predictCitationsByFrequency(target, dataPoints, sortedCitations);
        // 計算 Precision
        double prec = precision(target.citations, optResult.predictCitations);
        // 計算 Recall
        double rec = recall(target.citations, optResult.predictCitations);
        // 計算 Accuracy
        double acc = accuracy(target.citations, optResult.predictCitations);
        // 新增結果至 results
        ExperimentResult result;
        result.k = k;
        result.isK = true;  // 表示這是 k 值
        result.coefficient = optResult.coefficient;
        result.precision = prec;
        result.recall = rec;
        result.accuracy = acc;
        results.push_back(result);
    }
    return results;
}

// 定義實驗二：運用固定鄰居距離預測法條集(d方法)
std::vector<ExperimentResult> experiment2(
    const DataPoint& target,
    const std::vector<DataPoint>& dataPoints,
    double minD,
    double maxD,
    double step
) {
    // 定義回傳結果
    std::vector<ExperimentResult> results;
    // 定義 neighbors 以儲存鄰居
    std::vector<std::pair<int, double> > sorted_neighbors;
    std::vector<int> neighbors;
    // 排序鄰居
    sorted_neighbors = sortedNeighbors(dataPoints, target);
    // 依序處理每個 d 值
    for (double d = minD; d <= maxD; d += step) {
        // 找出距離在 d 以內的所有鄰居
        neighbors = findNeighborsWithinDistance(sorted_neighbors, d);
        // 統計 neighbors 中出現的法條與頻次
        std::unordered_map<Citation, int> citationCount = countCitationsInNeighbors(dataPoints, neighbors);
        // 依據 citationCount 排序法條與頻次
        std::vector<std::pair<Citation, int> > sortedCitations = sortCitationsByFrequency(citationCount);
        // 預測法條集
        OptimizationResult optResult = predictCitationsByFrequency(target, dataPoints, sortedCitations);
        // 計算 Precision
        double prec = precision(target.citations, optResult.predictCitations);
        // 計算 Recall
        double rec = recall(target.citations, optResult.predictCitations);
        // 計算 Accuracy
        double acc = accuracy(target.citations, optResult.predictCitations);
        // 新增結果至 results
        ExperimentResult result;
        result.d = d;
        result.isK = false;  // 表示這是 d 值
        result.coefficient = optResult.coefficient;
        result.precision = prec;
        result.recall = rec;
        result.accuracy = acc;
        results.push_back(result);
    }
    return results;
}
        
// 輸出實驗結果 - 使用 isK 判斷而不是 std::holds_alternative
void printResults(const std::vector<ExperimentResult>& results) {
    for (size_t i = 0; i < results.size(); ++i) {
        const ExperimentResult& result = results[i];
        // 使用 isK 來判斷是 k 還是 d
        if (result.isK) {
            std::cout << "k=" << result.k << ", Dice Coefficient=" << result.coefficient
                      << ", Precision=" << result.precision << ", Recall=" << result.recall
                      << ", Accuracy=" << result.accuracy << std::endl;
        } else {
            std::cout << "d=" << result.d << ", Dice Coefficient=" << result.coefficient
                      << ", Precision=" << result.precision << ", Recall=" << result.recall
                      << ", Accuracy=" << result.accuracy << std::endl;
        }
    }
}

// 主程式
int main() {
    // 讀取 CSV 檔案
    std::string filePath = "path/to/file.csv";
    // 測試檔
    // std::string filePath = "path/to/file.csv";
    std::vector<DataPoint> dataPoints = loadCSV(filePath); // 修正字串常量
    int k = 80;
    double minD = 0.4;
    double maxD = 0.8;
    double step = 0.01;
    // 隨機選取 randint 個 DataPoint 作為目標點
    int randint = 1000;

    std::vector<DataPoint> targetPoints;
    srand(time(NULL)); // 初始化隨機數生成器
    for (int i = 0; i < randint; i++) {
        int index = std::rand() % dataPoints.size();
        targetPoints.push_back(dataPoints[index]);
    }
    
    
    // 將結果依據 k 值或 d 值加總平均
    std::unordered_map<double, ExperimentResult> sumResults;
    std::unordered_map<double, int> countResults;
    // 定義 isolatedPoints 以儲存沒有鄰居的 DataPoint
    std::unordered_map<double, int> isolatedPoints;
    for (size_t i = 0; i < targetPoints.size(); ++i) {
        const DataPoint& target = targetPoints[i];
        // 實驗一
        std::vector<ExperimentResult> results1 = experiment1(target, dataPoints, k);
        for (size_t j = 0; j < results1.size(); ++j) {
            const ExperimentResult& result = results1[j];
            if (result.isK) {
                sumResults[result.k].isK = true;
                sumResults[result.k].coefficient += result.coefficient;
                sumResults[result.k].precision += result.precision;
                sumResults[result.k].recall += result.recall;
                sumResults[result.k].accuracy += result.accuracy;
                countResults[result.k]++;
            }
        }

        // 實驗二
        std::vector<ExperimentResult> results2 = experiment2(target, dataPoints, minD, maxD, step);
        for (size_t j = 0; j < results2.size(); ++j) {
            const ExperimentResult& result = results2[j];
            if (!result.isK) {
                sumResults[result.d].isK = false;
                sumResults[result.d].coefficient += result.coefficient;
                sumResults[result.d].precision += result.precision;
                sumResults[result.d].recall += result.recall;
                sumResults[result.d].accuracy += result.accuracy;
                countResults[result.d]++;
            }
            // 如果四個值都是 0, 則表示這是一個孤立點
            if (result.coefficient == 0 && result.precision == 0 && result.recall == 0 && result.accuracy == 0) {
                isolatedPoints[result.d]++;
            }
        }

    }
    // 輸出 k 平均結果
    for (std::unordered_map<double, ExperimentResult>::const_iterator it = sumResults.begin(); it != sumResults.end(); ++it) {
        double k = it->first;
        ExperimentResult avgResult = it->second;
        avgResult.coefficient /= countResults[k];
        avgResult.precision /= countResults[k];
        avgResult.recall /= countResults[k];
        avgResult.accuracy /= countResults[k];
        if (avgResult.isK) {
            std::cout << "Average result for k=" << k << ": Dice Coefficient=" << avgResult.coefficient
                      << ", Precision=" << avgResult.precision << ", Recall=" << avgResult.recall
                      << ", Accuracy=" << avgResult.accuracy << std::endl;
        }
    }

    // 輸出 d 平均結果
    for (std::unordered_map<double, ExperimentResult>::const_iterator it = sumResults.begin(); it != sumResults.end(); ++it) {
        double d = it->first;
        ExperimentResult avgResult = it->second;
        avgResult.coefficient /= countResults[d];
        avgResult.precision /= countResults[d];
        avgResult.recall /= countResults[d];
        avgResult.accuracy /= countResults[d];
        if (!avgResult.isK) {
            std::cout << "Average result for d=" << d << ": Dice Coefficient=" << avgResult.coefficient
                      << ", Precision=" << avgResult.precision << ", Recall=" << avgResult.recall
                      << ", Accuracy=" << avgResult.accuracy << ", Number of isolated points: " << isolatedPoints[d] << std::endl;
        }
    }

    return 0;
}