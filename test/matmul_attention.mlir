module {
  func.func @attention(
      %Q: tensor<512x512xf32>,
      %K: tensor<512x512xf32>,
      %V: tensor<512x512xf32>) -> tensor<512x512xf32> {
    // transformer.attention
    %cst_zero = arith.constant 0.0 : f32
    %cst_scale = arith.constant 0.044194173 : f32

    %scores_init = tensor.empty() : tensor<512x512xf32>
    %scores = linalg.fill ins(%cst_zero : f32) outs(%scores_init : tensor<512x512xf32>) -> tensor<512x512xf32>
    %qk = linalg.matmul ins(%Q, %K : tensor<512x512xf32>, tensor<512x512xf32>) outs(%scores : tensor<512x512xf32>) -> tensor<512x512xf32>

    %scaled = linalg.generic {
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
        ],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%qk : tensor<512x512xf32>)
      outs(%qk : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mul = arith.mulf %in, %cst_scale : f32
      linalg.yield %mul : f32
    } -> tensor<512x512xf32>

    %row_max_init = tensor.empty() : tensor<512xf32>
    %row_max_seed = linalg.fill ins(%cst_zero : f32) outs(%row_max_init : tensor<512xf32>) -> tensor<512xf32>
    %row_max = linalg.reduce ins(%scaled : tensor<512x512xf32>) outs(%row_max_seed : tensor<512xf32>) dimensions = [1] ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maximumf %lhs, %rhs : f32
      linalg.yield %max : f32
    }) -> tensor<512xf32>

    %shifted = linalg.generic {
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i)>,
          affine_map<(i, j) -> (i, j)>
        ],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%scaled, %row_max : tensor<512x512xf32>, tensor<512xf32>)
      outs(%scaled : tensor<512x512xf32>) {
    ^bb0(%score: f32, %max: f32, %out: f32):
      %sub = arith.subf %score, %max : f32
      linalg.yield %sub : f32
    } -> tensor<512x512xf32>

    %exp = linalg.generic {
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
        ],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%shifted : tensor<512x512xf32>)
      outs(%shifted : tensor<512x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %expv = math.exp %in : f32
      linalg.yield %expv : f32
    } -> tensor<512x512xf32>

    %row_sum_init = tensor.empty() : tensor<512xf32>
    %row_sum_seed = linalg.fill ins(%cst_zero : f32) outs(%row_sum_init : tensor<512xf32>) -> tensor<512xf32>
    %row_sum = linalg.reduce ins(%exp : tensor<512x512xf32>) outs(%row_sum_seed : tensor<512xf32>) dimensions = [1] ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    }) -> tensor<512xf32>

    %probs = linalg.generic {
        indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i)>,
          affine_map<(i, j) -> (i, j)>
        ],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%exp, %row_sum : tensor<512x512xf32>, tensor<512xf32>)
      outs(%exp : tensor<512x512xf32>) {
    ^bb0(%num: f32, %den: f32, %out: f32):
      %div = arith.divf %num, %den : f32
      linalg.yield %div : f32
    } -> tensor<512x512xf32>

    %out_init = tensor.empty() : tensor<512x512xf32>
    %out = linalg.fill ins(%cst_zero : f32) outs(%out_init : tensor<512x512xf32>) -> tensor<512x512xf32>
    %result = linalg.matmul ins(%probs, %V : tensor<512x512xf32>, tensor<512x512xf32>) outs(%out : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %result : tensor<512x512xf32>
  }
}
