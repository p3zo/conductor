const path = require('path');

const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');

const PRODUCTION = !!process.env.PRODUCTION;

module.exports = (env) => {
    const config = {
        mode: PRODUCTION ? 'production' : 'development',
        entry: {
            main: './src/index.js',
        },
        output: {
            path: path.resolve(__dirname, 'dist'),
            clean: true,
            filename: 'bundle.js',
        },
        devtool: 'inline-source-map',
        devServer: {
            headers: {
                // These two headers are required for cross origin isolation.
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp'
            },
            https: true,
            host: '0.0.0.0',
            port: 8090,
            historyApiFallback: {
                disableDotRule: true,
            },
            watchFiles: ['src/**/*'],
        },
        resolve: {
            extensions: ['.tsx', '.ts', '.js'],
        },
        module: {
            rules: [
                {
                    test: /\.wasm$/i,
                    type: 'javascript/auto',
                    use: [
                        {
                            loader: 'file-loader',
                        },
                    ],
                },
                {
                    test: /\.tsx?$/,
                    use: 'ts-loader',
                    exclude: /node_modules/,
                },
            ],
        },
        plugins: [
            new HtmlWebpackPlugin({
                template: 'index.html',
                filename: 'index.html',
            }),
            new CopyWebpackPlugin({
                patterns: [
                    'node_modules/@tensorflow/tfjs-backend-wasm/dist/*.wasm',
                    {from: 'node_modules/@handtracking.io/yoha/models/', to: './'},
                    {from: 'static', to: './'},
                ]
            })
        ],
        optimization: {
            minimizer: [new TerserPlugin({
                extractComments: false,
                exclude: /\.min\./,
            })],
        },
    };

    return [config];
}
