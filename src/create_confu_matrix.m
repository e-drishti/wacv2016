function create_confu_matrix(test_labels,predicted_labels)

num_categories = 3;
confusion_matrix zeros(num_categories, num_categories);
for i=1:length(predicted_categories)
    row = find(strcmp(test_labels{i}, categories));
    column = find(strcmp(predicted_labels{i}, categories));
    confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
end
end
