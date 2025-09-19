testDs != ls testDataSets/*.csv | sed s%testDataSets/%%

AllTexFiles=book.tex $(foreach dir,$(TexFolders),$(wildcard $(dir)/*.tex))



.PHONY : test fastTest clean
test:
	make $(foreach f,$(testDs), testResults/qMatrix_$(f))
	make $(foreach f,$(testDs), testResults/qMatrixTree_$(f))
	make $(foreach f,$(testDs), testResults/qMatrixTreeFast_$(f))

testResults/qMatrix_%: testDataSets/%
	test -d testResults || mkdir testResults
	python3 qMatrix.py "-o=$@" "$<" -i=patient_ids -i=id -debug "-oi=$@.pdf" "-op=$@_points.csv" > "$@.log" 2>&1

testResults/qMatrixTree_%: testDataSets/%
	test -d testResults || mkdir testResults
	python3 qMatrix.py "-o=$@" "$<" -i=patient_ids -i=id -tree -log -debug "-oi=$@.pdf" "-op=$@_points.csv" > "$@.log" 2>&1
  
testResults/qMatrixTreeFast_%: testDataSets/%
	test -d testResults || mkdir testResults
	python3 qMatrix.py "-o=$@" "$<" -i=patient_ids -i=id -tree-fast -log -debug "-oi=$@.pdf" "-op=$@_points.csv" > "$@.log" 2>&1


clean:
	-rm -r testResults

fastTest:
	# ===[ qMatrix ]==========================================================
	python3 qMatrix.py -i=patient_ids -i=id -o=- testDataSets/test_v1.csv
	#
	# ===[ qMatrix via tree (original version) ]==============================
	python3 qMatrix.py -i=patient_ids -i=id -tree -o=- testDataSets/test_v1.csv
	#
	# ===[ qMatrix via tree (fast version) ]==================================
	python3 qMatrix.py -i=patient_ids -i=id -tree-fast -o=- testDataSets/test_v1.csv
  
