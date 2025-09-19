testDs != ls testDataSets/*.csv | sed s%testDataSets/%%

AllTexFiles=book.tex $(foreach dir,$(TexFolders),$(wildcard $(dir)/*.tex))



.PHONY : test fastTest clean
test:
	make $(foreach f,$(testDs), testResults/$(f).ok)

testResults/%.csv.ok: testDataSets/%.csv
	$(eval q := testResults/$*)
	make "$q/brute-force.ok" "$q/tree.ok" "$q/tree-fast.ok"
	touch "$@"

testResults/%/brute-force.ok: testDataSets/%.csv
	$(eval p := testResults/$*/brute-force)
	mkdir -p "$p"
	$(eval outParam := "-o=$p/qMatrix.csv" "-oi=$p/projection.pdf" "-op=$p/projection.csv" "-od=$p/dendrogram.pdf" "-ou=$p/umap.pdf")
	$(eval config := -debug -i=patient_ids -i=id)
	python3 qMatrix.py $(config) $(outParam) "$<" 2>&1 | tee "$p/log.txt"
	touch "$@"

testResults/%/tree.ok: testDataSets/%.csv
	$(eval p := testResults/$*/tree)
	mkdir -p "$p"
	$(eval outParam := "-o=$p/qMatrix.csv" "-oi=$p/projection.pdf" "-op=$p/projection.csv" "-od=$p/dendrogram.pdf" "-ou=$p/umap.pdf")
	$(eval config := -tree -log -debug -i=patient_ids -i=id)
	python3 qMatrix.py $(config) $(outParam) "$<" 2>&1 | tee "$p/log.txt"
	touch "$@"
  
testResults/%/tree-fast.ok: testDataSets/%.csv
	$(eval p := testResults/$*/tree-fast)
	mkdir -p "$p"
	$(eval outParam := "-o=$p/qMatrix.csv" "-oi=$p/projection.pdf" "-op=$p/projection.csv" "-od=$p/dendrogram.pdf" "-ou=$p/umap.pdf")
	$(eval config := -tree-fast -log -debug -i=patient_ids -i=id)
	python3 qMatrix.py $(config) $(outParam) "$<" 2>&1 | tee "$p/log.txt"
	touch "$@"

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
  
