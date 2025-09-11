testDs != ls testDataSets/*.csv | sed s%testDataSets/%%

AllTexFiles=book.tex $(foreach dir,$(TexFolders),$(wildcard $(dir)/*.tex))



.PHONY : test
test:
	make $(foreach f,$(testDs), testResults/qMatrix_$(f))
	make $(foreach f,$(testDs), testResults/qMatrixTree_$(f))
	make $(foreach f,$(testDs), testResults/qMatrixTreeFast_$(f))

testResults/qMatrix_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id > "$@.log"

testResults/qMatrixTree_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id --tree --log > "$@.log"
  
testResults/qMatrixTreeFast_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id --tree-fast --log > "$@.log"
  
testResults/qMatrixTreeOriginal_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id --tree-original --log > "$@.log"
  
