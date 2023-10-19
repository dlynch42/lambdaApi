# Invoke lambda function
# Test 1
aws lambda invoke --function-name automodel --payload file://test/aapl.txt output/output_aapl.txt 

# Test 2
aws lambda invoke --function-name automodel --payload file://test/tsla.txt output/output_tsla.txt 

# Test 3
aws lambda invoke --function-name automodel --payload file://test/amzn.txt output/output_amzn.txt 