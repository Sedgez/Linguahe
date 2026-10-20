% Evaluate the emotion ANN predictions in emotion_test_results.csv.
% Place this script beside the CSV, then press Run in MATLAB.

scriptFolder = fileparts(mfilename('fullpath'));
resultsFile = fullfile(scriptFolder, 'emotion_test_results.csv');
T = readtable(resultsFile, 'TextType', 'string');

classNames = ["Angry", "Disgusted", "Happy", "Neutral", ...
              "Sad", "Scared", "Surprised"];
trueLabels = categorical(T.true_label, classNames);
predictedLabels = categorical(T.predicted_label, classNames);

C = confusionmat(trueLabels, predictedLabels, 'Order', categorical(classNames));
accuracy = sum(diag(C)) / max(sum(C(:)), 1);

truePositive = diag(C);
precision = truePositive ./ max(sum(C, 1)', 1);
recall = truePositive ./ max(sum(C, 2), 1);
f1 = 2 .* precision .* recall ./ max(precision + recall, eps);

classResults = table(classNames', sum(C, 2), precision, recall, f1, ...
    'VariableNames', {'Emotion', 'TestCount', 'Precision', 'Recall', 'F1'});
overallMetrics = [accuracy, mean(precision), mean(recall), mean(f1)];
metricNames = {'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-score'};

fprintf('Single-label accuracy: %.2f%% (%d/%d)\n', ...
    100 * accuracy, sum(diag(C)), sum(C(:)));
disp(classResults);

figure('Name', 'Emotion Classification — Single-Label Performance', ...
       'Color', 'white', 'Position', [100, 100, 1200, 800]);
bars = bar(overallMetrics, 0.52);
bars.FaceColor = 'flat';
bars.CData = [0.30, 0.45, 0.69; ...
              0.88, 0.47, 0.27; ...
              0.35, 0.65, 0.42; ...
              0.78, 0.30, 0.32];
ax = gca;
ax.XTick = 1:numel(metricNames);
ax.XTickLabel = metricNames;
ax.FontSize = 15;
ax.LineWidth = 1.2;
ax.Box = 'off';
ylim([0, 1.15]);
ylabel('Score', 'FontSize', 16);
title('Emotion Classification — Single-Label Performance', ...
      'FontSize', 23, 'FontWeight', 'bold');
text(1:numel(overallMetrics), overallMetrics + 0.018, ...
     compose('%.4f', overallMetrics), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
     'FontSize', 14, 'Color', 'black');
exportgraphics(gcf, fullfile(scriptFolder, ...
    'emotion_overall_performance.png'), 'Resolution', 200);

writetable(classResults, fullfile(scriptFolder, 'emotion_class_metrics.csv'));
