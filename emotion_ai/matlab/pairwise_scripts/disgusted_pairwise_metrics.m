%% Pairwise emotion metrics for disgusted vs each other emotion
% Reads the held-out predictions from ../emotion_test_results.csv.
% Pair accuracy requires an exact match to one of the two emotions.
% Precision, recall, and F1 are macro-averaged over the two pair classes.

clearvars;
clc;

scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder)
    scriptFolder = pwd;
end
candidatePaths = { ...
    fullfile(scriptFolder, 'emotion_test_results.csv'), ...
    fullfile(scriptFolder, '..', 'emotion_test_results.csv'), ...
    fullfile(pwd, 'emotion_test_results.csv')};
resultsFile = '';
for pathIndex = 1:numel(candidatePaths)
    if isfile(candidatePaths{pathIndex})
        resultsFile = candidatePaths{pathIndex};
        break;
    end
end
if isempty(resultsFile)
    [csvName, csvFolder] = uigetfile('*.csv', ...
        'Select emotion_test_results.csv');
    if isequal(csvName, 0)
        error('No results CSV was selected. Select emotion_test_results.csv and run again.');
    end
    resultsFile = fullfile(csvFolder, csvName);
end

T = readtable(resultsFile, 'TextType', 'string');
requiredColumns = ["true_label", "predicted_label"];
if ~all(ismember(requiredColumns, string(T.Properties.VariableNames)))
    error('The CSV must contain true_label and predicted_label columns.');
end

trueLabels = lower(strtrim(string(T.true_label)));
predictedLabels = lower(strtrim(string(T.predicted_label)));
trueLabels(trueLabels == "normal") = "neutral";
predictedLabels(predictedLabels == "normal") = "neutral";

classNames = ["angry", "disgusted", "happy", "neutral", ...
              "sad", "scared", "surprised"];
anchorEmotion = "disgusted";
comparisonEmotions = classNames(classNames ~= anchorEmotion);

numberOfPairs = numel(comparisonEmotions);
pairNames = strings(numberOfPairs, 1);
pairSampleCount = zeros(numberOfPairs, 1);
anchorSampleCount = zeros(numberOfPairs, 1);
comparisonSampleCount = zeros(numberOfPairs, 1);
accuracyPercent = nan(numberOfPairs, 1);
precisionPercent = nan(numberOfPairs, 1);
recallPercent = nan(numberOfPairs, 1);
f1ScorePercent = nan(numberOfPairs, 1);

for k = 1:numberOfPairs
    comparisonEmotion = comparisonEmotions(k);
    pairNames(k) = upper(anchorEmotion) + "-" + upper(comparisonEmotion);

    pairMask = (trueLabels == anchorEmotion) | ...
               (trueLabels == comparisonEmotion);
    pairTruth = trueLabels(pairMask);
    pairPrediction = predictedLabels(pairMask);

    isAnchorTruth = pairTruth == anchorEmotion;
    isComparisonTruth = pairTruth == comparisonEmotion;
    anchorTP = sum(isAnchorTruth & pairPrediction == anchorEmotion);
    anchorFP = sum(isComparisonTruth & pairPrediction == anchorEmotion);
    anchorFN = sum(isAnchorTruth & pairPrediction ~= anchorEmotion);
    comparisonTP = sum(isComparisonTruth & pairPrediction == comparisonEmotion);
    comparisonFP = sum(isAnchorTruth & pairPrediction == comparisonEmotion);
    comparisonFN = sum(isComparisonTruth & pairPrediction ~= comparisonEmotion);

    pairSampleCount(k) = numel(pairTruth);
    anchorSampleCount(k) = sum(isAnchorTruth);
    comparisonSampleCount(k) = sum(isComparisonTruth);

    if pairSampleCount(k) > 0
        accuracyPercent(k) = 100 * (anchorTP + comparisonTP) / pairSampleCount(k);

        anchorPrecision = anchorTP / max(anchorTP + anchorFP, 1);
        comparisonPrecision = comparisonTP / max(comparisonTP + comparisonFP, 1);
        anchorRecall = anchorTP / max(anchorTP + anchorFN, 1);
        comparisonRecall = comparisonTP / max(comparisonTP + comparisonFN, 1);
        anchorF1 = 2 * anchorPrecision * anchorRecall / ...
                   max(anchorPrecision + anchorRecall, eps);
        comparisonF1 = 2 * comparisonPrecision * comparisonRecall / ...
                       max(comparisonPrecision + comparisonRecall, eps);

        precisionPercent(k) = 100 * mean([anchorPrecision, comparisonPrecision]);
        recallPercent(k) = 100 * mean([anchorRecall, comparisonRecall]);
        f1ScorePercent(k) = 100 * mean([anchorF1, comparisonF1]);
    end
end

metricsTable = table(pairNames, pairSampleCount, anchorSampleCount, ...
    comparisonSampleCount, accuracyPercent, precisionPercent, ...
    recallPercent, f1ScorePercent, ...
    'VariableNames', {'EmotionPair', 'PairSampleCount', ...
    'AnchorSamples', 'ComparisonSamples', 'AccuracyPercent', ...
    'PrecisionPercent', 'RecallPercent', 'F1ScorePercent'});

outputCsv = fullfile(scriptFolder, 'disgusted_pairwise_metrics.csv');
writetable(metricsTable, outputCsv);
disp(metricsTable);

metricValues = [accuracyPercent, precisionPercent, ...
                recallPercent, f1ScorePercent];
figureHandle = figure('Color', 'white', 'Position', [100, 100, 1200, 650]);
barHandles = bar(metricValues, 'grouped');
ax = gca;
ax.XTick = 1:numberOfPairs;
ax.XTickLabel = pairNames;
ax.XTickLabelRotation = 35;
ax.FontSize = 11;
ax.Box = 'off';
ylim([0, 105]);
ylabel('Score (%)');
xlabel('Emotion pair');
title('disgusted Pairwise Emotion Classification Metrics', ...
      'FontWeight', 'bold');
legend({'Accuracy', 'Precision', 'Recall', 'F1-score'}, ...
       'Location', 'northoutside', 'Orientation', 'horizontal');
grid on;

% Show each metric value above its bar, similar to the reference graph.
for metricIndex = 1:numel(barHandles)
    barX = barHandles(metricIndex).XEndPoints;
    barY = barHandles(metricIndex).YEndPoints;
    valueLabels = compose('%.1f', barY);
    text(barX, barY + 1.2, valueLabels, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 9, ...
        'Color', [0.2, 0.2, 0.2]);
end

outputPng = fullfile(scriptFolder, 'disgusted_pairwise_metrics.png');
exportgraphics(figureHandle, outputPng, 'Resolution', 200);
fprintf('Saved table: %s\n', outputCsv);
fprintf('Saved chart: %s\n', outputPng);

