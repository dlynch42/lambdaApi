# Invoke lambda function via API Gateway
# Test 1
aws lambda invoke --function-name automodel --payload file://aapl.txt output/output_aapl.txt 

# Test 2
aws lambda invoke --function-name automodel --payload file://tsla.txt output/output_tsla.txt 

# Test 3
aws lambda invoke --function-name automodel --payload file://amzn.txt output/output_amzn.txt 