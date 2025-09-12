testDs != ls testDataSets/*.csv | sed s%testDataSets/%%

AllTexFiles=book.tex $(foreach dir,$(TexFolders),$(wildcard $(dir)/*.tex))



.PHONY : test
test:
	make $(foreach f,$(testDs), testResults/qMatrix_$(f))
	make $(foreach f,$(testDs), testResults/qMatrixTree_$(f))
	make $(foreach f,$(testDs), testResults/qMatrixTreeFast_$(f))

testResults/qMatrix_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id -oi "$@.pdf" -op "$@_points.csv" > "$@.log"

testResults/qMatrixTree_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id --tree --log -oi "$@.pdf" -op "$@_points.csv" > "$@.log"
  
testResults/qMatrixTreeFast_%: testDataSets/%
	test -d testResults || mkdir testResults
	time python3 qMatrix.py -o "$@" "$<" -i patient_ids -i id --tree-fast --log -oi "$@.pdf" -op "$@_points.csv" > "$@.log"


.PHONY : fastTest
fastTest:
	# ===[ qMatrix ]==========================================================
	python3 qMatrix.py -i patient_ids -i id --log -op - testDataSets/test_v1.csv
	#
	# ===[ qMatrix via tree (original version) ]==============================
	python3 qMatrix.py -i patient_ids -i id --tree --log -op - testDataSets/test_v1.csv
	#
	# ===[ qMatrix via tree (fast version) ]==================================
	python3 qMatrix.py -i patient_ids -i id --tree-fast --log -op - testDataSets/test_v1.csv
  
