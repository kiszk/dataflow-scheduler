// RUN: dataflow-scheduler-opt --ktir-legality-check --split-input-file --verify-diagnostics %s
#map = affine_map<(d0) -> (d0)>
// Only f16 <-> f32 has a device instruction: a conversion between any other
// widths is still rejected.
func.func @bf16_to_f32(%a: tensor<64xbf16>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xbf16>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: bf16, %out: f32):
    // expected-error @+1 {{V1 only supports add/mul/sub compute ops; found unsupported compute op}}
    %e = arith.extf %in : bf16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>
func.func @f64_to_f32(%a: tensor<64xf64>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf64>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: f64, %out: f32):
    // expected-error @+1 {{V1 only supports add/mul/sub compute ops; found unsupported compute op}}
    %t = arith.truncf %in : f64 to f32
    linalg.yield %t : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}
